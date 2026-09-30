#include "isvik/core/gguf_tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <optional>
#include <utility>

namespace isvik {
namespace {

constexpr std::string_view kSpaceMarker = "\xE2\x96\x81";

std::string PairKey(std::string_view left, std::string_view right) {
  std::string key;
  key.reserve(left.size() + right.size() + 1U);
  key.append(left);
  key.push_back('\0');
  key.append(right);
  return key;
}

std::vector<std::string> Utf8Symbols(std::string_view text) {
  std::vector<std::string> symbols;
  for (std::size_t index = 0; index < text.size();) {
    const unsigned char first = static_cast<unsigned char>(text[index]);
    std::size_t width = first < 0x80U ? 1U :
        (first & 0xe0U) == 0xc0U ? 2U :
        (first & 0xf0U) == 0xe0U ? 3U :
        (first & 0xf8U) == 0xf0U ? 4U : 1U;
    if (width > text.size() - index) width = 1U;
    bool valid = true;
    for (std::size_t offset = 1; offset < width; ++offset) {
      if ((static_cast<unsigned char>(text[index + offset]) & 0xc0U) != 0x80U) {
        valid = false;
        break;
      }
    }
    if (!valid) width = 1U;
    symbols.emplace_back(text.substr(index, width));
    index += width;
  }
  return symbols;
}

std::optional<uint8_t> ByteFallbackValue(std::string_view token) {
  if (token.size() != 6U || !token.starts_with("<0x") || token.back() != '>') {
    return std::nullopt;
  }
  const auto hex = [](char value) -> int {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
  };
  const int high = hex(token[3]);
  const int low = hex(token[4]);
  if (high < 0 || low < 0) return std::nullopt;
  return static_cast<uint8_t>((high << 4) | low);
}

std::string ByteFallbackToken(uint8_t value) {
  constexpr std::array<char, 16> digits{
      '0', '1', '2', '3', '4', '5', '6', '7',
      '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
  std::string token = "<0x00>";
  token[3] = digits[value >> 4U];
  token[4] = digits[value & 0x0fU];
  return token;
}

}  // namespace

StatusOr<GgufTokenizer> GgufTokenizer::Create(
    const GgufTokenizerConfig& config) {
  if (config.tokens.empty()) {
    return Status::InvalidArgument("GGUF has no embedded tokenizer vocabulary");
  }
  if (config.bos_token_id >= config.tokens.size() ||
      config.eos_token_id >= config.tokens.size() ||
      config.unknown_token_id >= config.tokens.size()) {
    return Status::InvalidArgument("GGUF tokenizer special token IDs are outside the vocabulary");
  }
  GgufTokenizer tokenizer;
  tokenizer.tokens_ = config.tokens;
  tokenizer.token_types_ = config.token_types;
  tokenizer.bos_token_id_ = config.bos_token_id;
  tokenizer.eos_token_id_ = config.eos_token_id;
  tokenizer.unknown_token_id_ = config.unknown_token_id;
  tokenizer.token_ids_.reserve(tokenizer.tokens_.size());
  for (std::size_t index = 0; index < tokenizer.tokens_.size(); ++index) {
    tokenizer.token_ids_.try_emplace(tokenizer.tokens_[index],
                                     static_cast<uint32_t>(index));
  }
  tokenizer.merge_ranks_.reserve(config.merges.size());
  for (std::size_t index = 0; index < config.merges.size(); ++index) {
    const std::string& merge = config.merges[index];
    const std::size_t separator = merge.find(' ');
    if (separator == std::string::npos || separator == 0U ||
        separator + 1U >= merge.size()) {
      continue;
    }
    tokenizer.merge_ranks_.try_emplace(
        PairKey(std::string_view(merge).substr(0U, separator),
                std::string_view(merge).substr(separator + 1U)),
        static_cast<uint32_t>(index));
  }
  return tokenizer;
}

StatusOr<std::vector<uint32_t>> GgufTokenizer::Encode(
    std::string_view text, bool add_bos) const {
  std::vector<uint32_t> result;
  if (add_bos) result.push_back(bos_token_id_);

  std::size_t cursor = 0U;
  while (cursor < text.size()) {
    std::size_t special_end = cursor;
    uint32_t special_id = 0U;
    bool special_found = false;
    if (text[cursor] == '<') {
      for (std::size_t end = text.find('>', cursor);
           end != std::string_view::npos && end - cursor <= 128U;
           end = text.find('>', end + 1U)) {
        const auto found = token_ids_.find(std::string(text.substr(cursor, end - cursor + 1U)));
        if (found != token_ids_.end()) {
          special_found = true;
          special_id = found->second;
          special_end = end + 1U;
          break;
        }
      }
    }
    if (special_found) {
      result.push_back(special_id);
      cursor = special_end;
      continue;
    }

    std::size_t plain_end = text.find('<', cursor);
    if (plain_end == std::string_view::npos) plain_end = text.size();
    if (plain_end == cursor) plain_end = std::min(text.size(), cursor + 1U);
    std::string normalized;
    normalized.reserve(plain_end - cursor + 8U);
    for (std::size_t index = cursor; index < plain_end; ++index) {
      if (text[index] == ' ') normalized.append(kSpaceMarker);
      else normalized.push_back(text[index]);
    }

    std::vector<std::string> symbols = Utf8Symbols(normalized);
    while (symbols.size() > 1U) {
      uint32_t best_rank = std::numeric_limits<uint32_t>::max();
      std::size_t best_index = symbols.size();
      for (std::size_t index = 0; index + 1U < symbols.size(); ++index) {
        const auto rank = merge_ranks_.find(PairKey(symbols[index], symbols[index + 1U]));
        if (rank != merge_ranks_.end() && rank->second < best_rank) {
          best_rank = rank->second;
          best_index = index;
        }
      }
      if (best_index == symbols.size()) break;
      symbols[best_index] += symbols[best_index + 1U];
      symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index + 1U));
    }
    for (const std::string& symbol : symbols) {
      const auto found = token_ids_.find(symbol);
      if (found != token_ids_.end()) {
        result.push_back(found->second);
        continue;
      }
      for (const unsigned char byte : symbol) {
        const auto byte_token = token_ids_.find(ByteFallbackToken(byte));
        result.push_back(byte_token == token_ids_.end() ? unknown_token_id_
                                                        : byte_token->second);
      }
    }
    cursor = plain_end;
  }
  return result;
}

std::string GgufTokenizer::DecodeToken(uint32_t token_id) const {
  if (token_id >= tokens_.size()) return {};
  if (token_id < token_types_.size() && token_types_[token_id] == 3) return {};
  const std::string& token = tokens_[token_id];
  if (const auto byte = ByteFallbackValue(token); byte.has_value()) {
    return std::string(1U, static_cast<char>(*byte));
  }
  std::string decoded;
  for (std::size_t index = 0; index < token.size();) {
    if (std::string_view(token).substr(index).starts_with(kSpaceMarker)) {
      decoded.push_back(' ');
      index += kSpaceMarker.size();
    } else {
      decoded.push_back(token[index++]);
    }
  }
  return decoded;
}

std::string GgufTokenizer::Decode(const std::vector<uint32_t>& token_ids) const {
  std::string result;
  for (const uint32_t token_id : token_ids) result += DecodeToken(token_id);
  return result;
}

}  // namespace isvik
