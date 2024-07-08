#include <iostream>
#include <vector>
#include <array>
#include <tuple>
#include <algorithm>
#include <gtest/gtest.h>

// Helper function to interleave the perm array
std::vector<int> interleave_perms(const std::vector<int>& perm) {
    std::vector<int> interleaved_perm;
    std::array<int, 8> interleave = {0, 2, 4, 6, 1, 3, 5, 7};

    int num_rows = perm.size() / 8;
    for (int i = 0; i < num_rows; ++i) {
        std::array<int, 8> row;
        std::copy(perm.begin() + i * 8, perm.begin() + (i + 1) * 8, row.begin());
        for (int j : interleave) {
            interleaved_perm.push_back(row[j]);
        }
    }
    
    return interleaved_perm;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> get_perms() {
    std::vector<int> perm;

    for (int i = 0; i < 32; ++i) {
        std::vector<int> perm1;
        int col = i / 4;
        for (int block : {0, 1}) {
            for (int row : {
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1
            }) {
                perm1.push_back(16 * row + col + 8 * block);
            }
        }
        for (int j = 0; j < 4; ++j) {
            for (int p : perm1) {
                perm.push_back(p + 256 * j);
            }
        }
    }

    // Interleave the perm array
    perm = interleave_perms(perm);
    
    std::vector<int> scale_perm;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            scale_perm.push_back(i + 8 * j);
        }
    }

    std::vector<int> scale_perm_single;
    for (int i = 0; i < 4; ++i) {
        for (int j : {0, 1, 8, 9, 16, 17, 24, 25}) {
            scale_perm_single.push_back(2 * i + j);
        }
    }

    return std::make_tuple(perm, scale_perm, scale_perm_single);
}

TEST(EfficientI4MatmulTest, ParamPermutate)
{
    auto [perm, scale_perm, scale_perm_single] = get_perms();
    
    std::cout << "perm: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << perm[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "scale_perm: ";
    for (const auto& val : scale_perm) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "scale_perm_single: ";
    for (const auto& val : scale_perm_single) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
