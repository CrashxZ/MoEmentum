// Narrow compatibility parser derived from the deployed calibration-v4 header.
// Frozen geometry includes the inactive MTP layer. This is not a map fitter.
// Validate dimensions BEFORE allocation; no private map bytes or paths belong here.
#pragma once
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace q36_calibration {

struct CalibrationMap {
    std::string model_sha256;
    size_t n_layers = 0;
    size_t n_experts = 0;
    std::vector<std::vector<int32_t>> layer_targets; // [li][slot_idx] -> expert_id
};

inline bool is_valid_hex_sha256(const std::string& s) {
    if (s.size() != 64) return false;
    for (char c : s) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

inline std::string to_lower(const std::string& s) {
    std::string out = s;
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

inline bool load_calibration_map_file(const std::string& filepath, CalibrationMap& out_map, std::string& err_msg) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        err_msg = "could not open calibration map file: " + filepath;
        return false;
    }

    std::string line;
    // Helper to get next non-empty trimmed line
    auto get_next_line = [&](std::string& l) -> bool {
        while (std::getline(in, l)) {
            size_t start = l.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;
            size_t end = l.find_last_not_of(" \t\r\n");
            l = l.substr(start, end - start + 1);
            return true;
        }
        return false;
    };

    // Line 1: magic version
    if (!get_next_line(line)) {
        err_msg = "empty map file";
        return false;
    }
    if (line != "STRATA_CALIBRATION_MAP_V1") {
        err_msg = "invalid magic version in map file: " + line;
        return false;
    }

    // Line 2: model_sha256
    if (!get_next_line(line)) {
        err_msg = "truncated map file: missing model_sha256";
        return false;
    }
    {
        std::istringstream iss(line);
        std::string tag, sha;
        if (!(iss >> tag >> sha) || tag != "model_sha256") {
            err_msg = "malformed model_sha256 line: " + line;
            return false;
        }
        std::string extra;
        if (iss >> extra) {
            err_msg = "extra trailing tokens on model_sha256 line: " + line;
            return false;
        }
        if (!is_valid_hex_sha256(sha)) {
            err_msg = "invalid model_sha256 format (must be 64 hex characters): " + sha;
            return false;
        }
        out_map.model_sha256 = sha;
    }

    // Line 3: layers <N>
    if (!get_next_line(line)) {
        err_msg = "truncated map file: missing layers header";
        return false;
    }
    {
        std::istringstream iss(line);
        std::string tag;
        long long nl = 0;
        if (!(iss >> tag >> nl) || tag != "layers" || nl != 41) {
            err_msg = "unsupported frozen layers count (requires 41): " + line;
            return false;
        }
        std::string extra;
        if (iss >> extra) {
            err_msg = "extra tokens on layers line: " + line;
            return false;
        }
        out_map.n_layers = static_cast<size_t>(nl);
    }

    // Line 4: experts <N>
    if (!get_next_line(line)) {
        err_msg = "truncated map file: missing experts header";
        return false;
    }
    {
        std::istringstream iss(line);
        std::string tag;
        long long ne = 0;
        if (!(iss >> tag >> ne) || tag != "experts" || ne != 256) {
            err_msg = "unsupported frozen experts count (requires 256): " + line;
            return false;
        }
        std::string extra;
        if (iss >> extra) {
            err_msg = "extra tokens on experts line: " + line;
            return false;
        }
        out_map.n_experts = static_cast<size_t>(ne);
    }

    out_map.layer_targets.assign(out_map.n_layers, std::vector<int32_t>());

    // Read per-layer lines
    for (size_t li = 0; li < out_map.n_layers; ++li) {
        if (!get_next_line(line)) {
            err_msg = "truncated map file: expected " + std::to_string(out_map.n_layers) +
                      " layers, got " + std::to_string(li);
            return false;
        }
        std::istringstream iss(line);
        std::string tag_l, tag_s, tag_e;
        long long read_li = -1, read_slots = -1;
        if (!(iss >> tag_l >> read_li >> tag_s >> read_slots >> tag_e) ||
            tag_l != "layer" || tag_s != "slots" || tag_e != "experts") {
            err_msg = "malformed layer line format at line: " + line;
            return false;
        }
        if (read_li != static_cast<long long>(li)) {
            err_msg = "layer index mismatch: expected " + std::to_string(li) + ", got " + std::to_string(read_li);
            return false;
        }
        if (read_slots != 23) {
            err_msg = "invalid slot count " + std::to_string(read_slots) + " at layer " + std::to_string(li);
            return false;
        }
        size_t slots = static_cast<size_t>(read_slots);
        std::vector<int32_t> targets;
        targets.reserve(slots);
        std::vector<bool> seen(out_map.n_experts, false);

        for (size_t s = 0; s < slots; ++s) {
            long long exp_id = -1;
            if (!(iss >> exp_id)) {
                err_msg = "fewer expert IDs than slots (" + std::to_string(slots) + ") at layer " + std::to_string(li);
                return false;
            }
            if (exp_id < 0 || static_cast<size_t>(exp_id) >= out_map.n_experts) {
                err_msg = "expert ID " + std::to_string(exp_id) + " out of range [0, " +
                          std::to_string(out_map.n_experts - 1) + "] at layer " + std::to_string(li);
                return false;
            }
            if (seen[static_cast<size_t>(exp_id)]) {
                err_msg = "duplicate expert ID " + std::to_string(exp_id) + " at layer " + std::to_string(li);
                return false;
            }
            seen[static_cast<size_t>(exp_id)] = true;
            targets.push_back(static_cast<int32_t>(exp_id));
        }

        std::string extra;
        if (iss >> extra) {
            err_msg = "extra trailing tokens on layer " + std::to_string(li) + " line: " + line;
            return false;
        }

        out_map.layer_targets[li] = std::move(targets);
    }

    // Trailing non-empty lines
    if (get_next_line(line)) {
        err_msg = "trailing unexpected data after layers: " + line;
        return false;
    }

    return true;
}

inline bool validate_topology(const CalibrationMap& map, size_t actual_layers, size_t actual_experts,
                              const std::vector<size_t>& actual_layer_slots, std::string& err_msg) {
    if (map.n_layers != actual_layers) {
        err_msg = "layer count mismatch: map has " + std::to_string(map.n_layers) +
                  ", engine has " + std::to_string(actual_layers);
        return false;
    }
    if (map.n_experts != actual_experts) {
        err_msg = "expert count mismatch: map has " + std::to_string(map.n_experts) +
                  ", engine has " + std::to_string(actual_experts);
        return false;
    }
    if (actual_layer_slots.size() != actual_layers) {
        err_msg = "actual layer slots size mismatch: expected " + std::to_string(actual_layers) +
                  ", got " + std::to_string(actual_layer_slots.size());
        return false;
    }
    for (size_t li = 0; li < actual_layers; ++li) {
        size_t expected_slots = actual_layer_slots[li];
        {
            if (map.layer_targets[li].size() != expected_slots) {
                err_msg = "slot count mismatch at layer " + std::to_string(li) + ": map has " +
                          std::to_string(map.layer_targets[li].size()) + ", engine requires " +
                          std::to_string(expected_slots);
                return false;
            }
        }
    }
    return true;
}

inline bool verify_model_sha256(const CalibrationMap& map, const std::string& expected_sha256, std::string& err_msg) {
    if (!is_valid_hex_sha256(expected_sha256)) { err_msg = "invalid expected model SHA256"; return false; }
    const std::string& norm_exp = expected_sha256;
    if (map.model_sha256 != norm_exp) {
        err_msg = "model SHA256 mismatch: map declared " + map.model_sha256 +
                  ", expected " + norm_exp;
        return false;
    }
    return true;
}

} // namespace q36_calibration
