#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "finetune_ops/core/tokenizer_bpe.h"

using namespace ops;

namespace {

size_t skip_ws(const std::string& s, size_t pos) {
    while (pos < s.size()) {
        const char c = s[pos];
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
        ++pos;
    }
    return pos;
}

bool parse_json_string_literal(const std::string& s,
                               size_t quote_start,
                               std::string& out,
                               size_t* next_pos) {
    if (quote_start >= s.size() || s[quote_start] != '"') return false;
    std::string value;
    bool escape = false;
    for (size_t i = quote_start + 1; i < s.size(); ++i) {
        const char c = s[i];
        if (escape) {
            switch (c) {
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                default: value.push_back(c); break;
            }
            escape = false;
            continue;
        }
        if (c == '\\') {
            escape = true;
            continue;
        }
        if (c == '"') {
            out = std::move(value);
            if (next_pos) *next_pos = i + 1;
            return true;
        }
        value.push_back(c);
    }
    return false;
}

bool extract_json_string_field(const std::string& line,
                               const std::string& key,
                               std::string& out) {
    const std::string needle = "\"" + key + "\"";
    size_t key_pos = line.find(needle);
    if (key_pos == std::string::npos) return false;
    size_t colon = line.find(':', key_pos + needle.size());
    if (colon == std::string::npos) return false;
    size_t value_pos = skip_ws(line, colon + 1);
    return parse_json_string_literal(line, value_pos, out, nullptr);
}

struct Row {
    int line_no = 0;
    std::string prompt;
    std::string chosen;
    std::string rejected;
};

bool parse_row(const std::string& line, int line_no, Row& row) {
    row.line_no = line_no;
    return extract_json_string_field(line, "prompt", row.prompt) &&
           extract_json_string_field(line, "chosen", row.chosen) &&
           extract_json_string_field(line, "rejected", row.rejected);
}

int response_tokens_after_concat(const std::vector<int>& prompt_ids,
                                 const std::vector<int>& response_ids,
                                 int seq_len) {
    if (static_cast<int>(prompt_ids.size()) >= seq_len) return 0;
    return std::min<int>(response_ids.size(), seq_len - static_cast<int>(prompt_ids.size()));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pretrained_dir> <jsonl_path> <seq_len> [row_numbers...]\n";
        return 1;
    }

    const std::string pretrained_dir = argv[1];
    const std::string jsonl_path = argv[2];
    const int seq_len = std::stoi(argv[3]);
    std::set<int> focus_rows;
    for (int i = 4; i < argc; ++i) {
        focus_rows.insert(std::stoi(argv[i]));
    }

    GPT2BPETokenizer tok(BPEConfig::from_pretrained(pretrained_dir));
    tok.load();

    std::ifstream fin(jsonl_path);
    if (!fin.is_open()) {
        std::cerr << "Cannot open " << jsonl_path << "\n";
        return 1;
    }

    int total = 0;
    int kept = 0;
    int skipped_parse = 0;
    int skipped_no_response = 0;
    std::vector<std::tuple<int, int, int, int, int>> borderline;
    std::vector<int> skipped_rows;

    std::string line;
    int line_no = 0;
    while (std::getline(fin, line)) {
        ++line_no;
        if (line.empty()) continue;
        ++total;
        Row row;
        if (!parse_row(line, line_no, row)) {
            skipped_parse++;
            continue;
        }

        auto prompt_ids = tok.encode(row.prompt, false, 0, true);
        auto chosen_ids = tok.encode(row.chosen, false, 0, true);
        auto rejected_ids = tok.encode(row.rejected, false, 0, true);

        const int chosen_resp = response_tokens_after_concat(prompt_ids, chosen_ids, seq_len);
        const int rejected_resp = response_tokens_after_concat(prompt_ids, rejected_ids, seq_len);
        const int prompt_len = static_cast<int>(prompt_ids.size());

        if (focus_rows.empty() || focus_rows.count(line_no) > 0) {
            auto prompt_tokens = tok.encode(row.prompt, false, 0, true);
            std::cout << "ROW " << line_no
                      << " prompt_len=" << prompt_len
                      << " chosen_len=" << chosen_ids.size()
                      << " rejected_len=" << rejected_ids.size()
                      << " chosen_resp=" << chosen_resp
                      << " rejected_resp=" << rejected_resp
                      << " slack=" << (seq_len - prompt_len)
                      << " prompt_chars=" << row.prompt.size()
                      << " chosen_chars=" << row.chosen.size()
                      << " rejected_chars=" << row.rejected.size()
                      << "\n";
            std::cout << "  prompt_last_tokens:";
            const int start = std::max<int>(0, static_cast<int>(prompt_tokens.size()) - 20);
            for (int i = start; i < static_cast<int>(prompt_tokens.size()); ++i) {
                std::cout << " " << prompt_tokens[i];
            }
            std::cout << "\n";
            std::cout << "  prompt_all_tokens:";
            for (int id : prompt_tokens) {
                std::cout << " " << id;
            }
            std::cout << "\n";
            std::cout << "  prompt_tail_text: "
                      << row.prompt.substr(row.prompt.size() > 180 ? row.prompt.size() - 180 : 0)
                      << "\n";
        }

        if (chosen_resp == 0 || rejected_resp == 0) {
            skipped_no_response++;
            skipped_rows.push_back(line_no);
            borderline.emplace_back(prompt_len, chosen_resp, rejected_resp, line_no, 0);
            continue;
        }

        kept++;
        const int slack = seq_len - prompt_len;
        if (slack <= 4) {
            borderline.emplace_back(prompt_len, chosen_resp, rejected_resp, line_no, slack);
        }
    }

    std::sort(borderline.begin(), borderline.end(), std::greater<>());

    std::cout << "total=" << total
              << " kept=" << kept
              << " skipped_parse=" << skipped_parse
              << " skipped_no_response=" << skipped_no_response
              << " seq_len=" << seq_len
              << "\n";

    std::cout << "Top borderline rows (prompt_len, chosen_resp, rejected_resp, line_no, slack):\n";
    for (size_t i = 0; i < std::min<size_t>(borderline.size(), 20); ++i) {
        const auto& [prompt_len, chosen_resp, rejected_resp, row_no, slack] = borderline[i];
        std::cout << prompt_len << " " << chosen_resp << " " << rejected_resp << " " << row_no << " " << slack << "\n";
    }

    std::cout << "Skipped row numbers:\n";
    for (size_t i = 0; i < skipped_rows.size(); ++i) {
        if (i) std::cout << " ";
        std::cout << skipped_rows[i];
    }
    std::cout << "\n";

    return 0;
}
