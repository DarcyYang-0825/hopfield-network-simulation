#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define BASE_N 25
#define MAX_PATTERNS 26
#define MAX_STORED 15
#define BATCH_PER_LETTER 50

// 26个字母的基础图案 (5×5)
int base_patterns[26][25] = {
    // A
    {-1,-1,+1,-1,-1, -1,+1,-1,+1,-1, +1,+1,+1,+1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1},
    // B
    {+1,+1,+1,+1,-1, +1,-1,-1,-1,+1, +1,+1,+1,+1,-1, +1,-1,-1,-1,+1, +1,+1,+1,+1,-1},
    // C
    {+1,+1,+1,+1,+1, +1,-1,-1,-1,-1, +1,-1,-1,-1,-1, +1,-1,-1,-1,-1, +1,+1,+1,+1,+1},
    // D
    {+1,+1,+1,+1,-1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,-1},
    // E
    {+1,+1,+1,+1,+1, +1,-1,-1,-1,-1, +1,+1,+1,+1,-1, +1,-1,-1,-1,-1, +1,+1,+1,+1,+1},
    // F
    {+1,+1,+1,+1,+1, +1,-1,-1,-1,-1, +1,+1,+1,+1,-1, +1,-1,-1,-1,-1, +1,-1,-1,-1,-1},
    // G
    {+1,+1,+1,+1,+1, +1,-1,-1,-1,-1, +1,-1,+1,+1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,+1},
    // H
    {+1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1},
    // I
    {+1,+1,+1,+1,+1, -1,-1,+1,-1,-1, -1,-1,+1,-1,-1, -1,-1,+1,-1,-1, +1,+1,+1,+1,+1},
    // J
    {-1,-1,-1,-1,+1, -1,-1,-1,-1,+1, -1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,+1},
    // K
    {+1,-1,-1,-1,+1, +1,-1,-1,+1,-1, +1,+1,+1,-1,-1, +1,-1,-1,+1,-1, +1,-1,-1,-1,+1},
    // L
    {+1,-1,-1,-1,-1, +1,-1,-1,-1,-1, +1,-1,-1,-1,-1, +1,-1,-1,-1,-1, +1,+1,+1,+1,+1},
    // M
    {+1,-1,-1,-1,+1, +1,+1,-1,+1,+1, +1,-1,+1,-1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1},
    // N
    {+1,-1,-1,-1,+1, +1,+1,-1,-1,+1, +1,-1,+1,-1,+1, +1,-1,-1,+1,+1, +1,-1,-1,-1,+1},
    // O
    {+1,+1,+1,+1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,+1},
    // P
    {+1,+1,+1,+1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,+1, +1,-1,-1,-1,-1, +1,-1,-1,-1,-1},
    // Q
    {+1,+1,+1,+1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,-1,-1,+1,+1, +1,+1,+1,+1,+1},
    // R
    {+1,+1,+1,+1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,+1, +1,-1,-1,+1,-1, +1,-1,-1,-1,+1},
    // S
    {+1,+1,+1,+1,+1, +1,-1,-1,-1,-1, +1,+1,+1,+1,+1, -1,-1,-1,-1,+1, +1,+1,+1,+1,+1},
    // T
    {+1,+1,+1,+1,+1, -1,-1,+1,-1,-1, -1,-1,+1,-1,-1, -1,-1,+1,-1,-1, -1,-1,+1,-1,-1},
    // U
    {+1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,+1,+1,+1,+1},
    // V
    {+1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,-1,-1,-1,+1, -1,+1,-1,+1,-1, -1,-1,+1,-1,-1},
    // W
    {+1,-1,-1,-1,+1, +1,-1,-1,-1,+1, +1,-1,+1,-1,+1, +1,+1,-1,+1,+1, +1,-1,-1,-1,+1},
    // X
    {+1,-1,-1,-1,+1, -1,+1,-1,+1,-1, -1,-1,+1,-1,-1, -1,+1,-1,+1,-1, +1,-1,-1,-1,+1},
    // Y
    {+1,-1,-1,-1,+1, -1,+1,-1,+1,-1, -1,-1,+1,-1,-1, -1,-1,+1,-1,-1, -1,-1,+1,-1,-1},
    // Z
    {+1,+1,+1,+1,+1, -1,-1,-1,+1,-1, -1,-1,+1,-1,-1, -1,+1,-1,-1,-1, +1,+1,+1,+1,+1}
};

// 动态网络结构
typedef struct {
    int N;
    double** J;
    int* S;
} DynamicNetwork;

DynamicNetwork net;
int current_N = BASE_N;

// ========== 双线性插值算法（大尺寸用） ==========

// 生成测试图像（对称双线性插值）- 用于大尺寸
int* generate_test_letter_bilinear(char letter, int width, int height, float noise_rate) {
    if (width <= 0 || height <= 0) return NULL;
    
    int* image = (int*)malloc(width * height * sizeof(int));
    if (!image) return NULL;
    
    int base_idx = letter - 'A';
    float center_x = (width - 1) / 2.0f;
    float center_y = (height - 1) / 2.0f;
    float scale_x = 4.0f / (width - 1);
    float scale_y = 4.0f / (height - 1);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float src_x = (x - center_x) * scale_x + 2.0f;
            float src_y = (y - center_y) * scale_y + 2.0f;
            
            if (src_x < 0.0f) src_x = 0.0f;
            if (src_x > 4.0f) src_x = 4.0f;
            if (src_y < 0.0f) src_y = 0.0f;
            if (src_y > 4.0f) src_y = 4.0f;
            
            int x1 = (int)src_x;
            int y1 = (int)src_y;
            int x2 = x1 + 1;
            int y2 = y1 + 1;
            
            if (x2 > 4) x2 = 4;
            if (y2 > 4) y2 = 4;
            
            float wx = src_x - x1;
            float wy = src_y - y1;
            
            float v11 = (base_patterns[base_idx][y1*5 + x1] == +1) ? 1.0f : 0.0f;
            float v12 = (base_patterns[base_idx][y1*5 + x2] == +1) ? 1.0f : 0.0f;
            float v21 = (base_patterns[base_idx][y2*5 + x1] == +1) ? 1.0f : 0.0f;
            float v22 = (base_patterns[base_idx][y2*5 + x2] == +1) ? 1.0f : 0.0f;
            
            float interpolated = v11 * (1-wx) * (1-wy) + v12 * wx * (1-wy) + 
                               v21 * (1-wx) * wy + v22 * wx * wy;
            
            int pixel_val = (interpolated > 0.5f) ? +1 : -1;
            
            if ((float)rand() / RAND_MAX < noise_rate) {
                pixel_val = -pixel_val;
            }
            
            image[y * width + x] = pixel_val;
        }
    }
    
    return image;
}

// ========== 过采样算法（小尺寸用） ==========

// 生成测试图像（最近邻复制）- 用于小尺寸的过采样
int* generate_test_letter_nearest(char letter, int width, int height, float noise_rate) {
    if (width <= 0 || height <= 0) return NULL;
    
    int* image = (int*)malloc(width * height * sizeof(int));
    if (!image) return NULL;
    
    int base_idx = letter - 'A';
    
    // 简单映射：每个像素取最近的5×5位置
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int src_x = (x * 5) / width;
            int src_y = (y * 5) / height;
            
            if (src_x >= 5) src_x = 4;
            if (src_y >= 5) src_y = 4;
            
            int pixel_val = base_patterns[base_idx][src_y * 5 + src_x];
            
            if ((float)rand() / RAND_MAX < noise_rate) {
                pixel_val = -pixel_val;
            }
            
            image[y * width + x] = pixel_val;
        }
    }
    
    return image;
}

// 过采样标准化到5×5
int* standardize_to_5x5_oversampling(int* input, int width, int height) {
    int* output = (int*)malloc(25 * sizeof(int));
    if (!output) return NULL;
    
    // 计算放大倍数（确保5×5每个格子对应整数个像素）
    int expand_factor = 5;
    int expanded_w = width * expand_factor;
    int expanded_h = height * expand_factor;
    
    // 第一步：放大图像（每个像素复制成expand_factor×expand_factor块）
    int* expanded = (int*)malloc(expanded_w * expanded_h * sizeof(int));
    if (!expanded) {
        free(output);
        return NULL;
    }
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int value = input[y * width + x];
            for (int dy = 0; dy < expand_factor; dy++) {
                for (int dx = 0; dx < expand_factor; dx++) {
                    expanded[(y * expand_factor + dy) * expanded_w + (x * expand_factor + dx)] = value;
                }
            }
        }
    }
    
    // 第二步：缩小到5×5，每个格子统计多数
    int cell_w = expanded_w / 5;
    int cell_h = expanded_h / 5;
    
    for (int ty = 0; ty < 5; ty++) {
        for (int tx = 0; tx < 5; tx++) {
            int plus_count = 0;
            int total = 0;
            
            for (int dy = 0; dy < cell_h; dy++) {
                for (int dx = 0; dx < cell_w; dx++) {
                    int idx = (ty * cell_h + dy) * expanded_w + (tx * cell_w + dx);
                    if (expanded[idx] == +1) plus_count++;
                    total++;
                }
            }
            
            output[ty * 5 + tx] = (plus_count > total / 2) ? +1 : -1;
        }
    }
    
    free(expanded);
    return output;
}

// ========== 网络函数 ==========

// 初始化网络
void init_network(int N) {
    current_N = N;
    net.N = N;
    net.J = (double**)malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        net.J[i] = (double*)calloc(N, sizeof(double));
    }
    net.S = (int*)malloc(N * sizeof(int));
}

// 释放网络
void free_network() {
    if (net.J) {
        for (int i = 0; i < current_N; i++) free(net.J[i]);
        free(net.J);
    }
    if (net.S) free(net.S);
}

// 训练网络
void train_network(int* indices, int count) {
    int N = net.N;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            net.J[i][j] = 0.0;
    
    for (int mu = 0; mu < count; mu++) {
        int idx = indices[mu];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (i != j)
                    net.J[i][j] += (double)(base_patterns[idx][i] * base_patterns[idx][j]);
    }
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            net.J[i][j] /= N;
}

// 异步更新
void update_async() {
    int N = net.N;
    int* order = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) order[i] = i;
    for (int i = N-1; i > 0; i--) {
        int j = rand() % (i+1);
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }
    for (int k = 0; k < N; k++) {
        int i = order[k];
        double sum = 0.0;
        for (int j = 0; j < N; j++)
            if (i != j) sum += net.J[i][j] * net.S[j];
        net.S[i] = (sum >= 0) ? +1 : -1;
    }
    free(order);
}

// 检查收敛
int is_converged(int prev[]) {
    for (int i = 0; i < net.N; i++)
        if (net.S[i] != prev[i]) return 0;
    return 1;
}

// 处理输入并恢复（使用过采样标准化）
int process_input_oversampling(int* input_pattern, int width, int height, int target_idx) {
    int* standardized = standardize_to_5x5_oversampling(input_pattern, width, height);
    if (!standardized) return 0;
    
    for (int i = 0; i < net.N; i++)
        net.S[i] = standardized[i];
    
    int* prev = (int*)malloc(net.N * sizeof(int));
    int steps = 0;
    
    while (steps < 50) {
        for (int i = 0; i < net.N; i++) prev[i] = net.S[i];
        update_async();
        steps++;
        if (is_converged(prev)) break;
    }
    
    int success = 1;
    for (int i = 0; i < net.N; i++) {
        if (net.S[i] != base_patterns[target_idx][i]) {
            success = 0;
            break;
        }
    }
    
    free(standardized);
    free(prev);
    return success;
}

// ========== 热力图（过采样版本，3×3到15×15） ==========

void test_heatmap_oversampling(int* store_indices, int store_count) {
    printf("\n╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║              热力图分析 - 过采样算法（3×3 到 15×15）                        ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ 存储字母数: %-3d  每组合测试: %-3d次/字母                                     ║\n", 
           store_count, BATCH_PER_LETTER);
    printf("║ 算法: 过采样多数投票法                                                       ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n正在测试，请稍候...\n\n");
    
    // 存储结果：5个噪声级别 × 13个尺寸(3-15)
    float results[5][13];
    
    // 测试所有组合
    for (int n = 0; n < 5; n++) {
        float noise_rate = (n + 1) * 0.1f;
        
        for (int s = 0; s < 13; s++) {
            int size = s + 3;  // 3到15
            int total_success = 0;
            int total_tests = store_count * BATCH_PER_LETTER;
            
            for (int i = 0; i < store_count; i++) {
                int letter_idx = store_indices[i];
                char letter = 'A' + letter_idx;
                
                for (int t = 0; t < BATCH_PER_LETTER; t++) {
                    // 使用最近邻生成测试图像
                    int* test_image = generate_test_letter_nearest(letter, size, size, noise_rate);
                    if (!test_image) continue;
                    
                    // 使用过采样标准化
                    if (process_input_oversampling(test_image, size, size, letter_idx))
                        total_success++;
                    
                    free(test_image);
                }
            }
            
            results[n][s] = total_success * 100.0f / total_tests;
        }
        printf("  噪声 %.0f%% 测试完成\n", noise_rate * 100);
    }
    
    printf("\n");
    
    // 打印彩色热力图
    printf("╔══════╦══════════════════════════════════════════════════════════════════╗\n");
    printf("║噪声%% ║                        图像尺寸                                  ║\n");
    printf("╠══════╬");
    for (int size = 3; size <= 15; size++)
        printf("═══");
    printf("╣\n");
    
    // 列标题
    printf("║      ║");
    for (int size = 3; size <= 15; size++)
        printf("%2d ", size);
    printf("║\n");
    
    printf("╠══════╬");
    for (int size = 3; size <= 15; size++)
        printf("═══");
    printf("╣\n");
    
    // 数据行
    for (int n = 0; n < 5; n++) {
        printf("║ %3.0f%% ║", (n + 1) * 10.0f);
        
        for (int s = 0; s < 13; s++) {
            float rate = results[n][s];
            
            // 根据成功率选择颜色
            if (rate >= 95) printf("\033[48;2;0;150;0m");
            else if (rate >= 85) printf("\033[48;2;34;139;34m");
            else if (rate >= 75) printf("\033[48;2;50;205;50m");
            else if (rate >= 65) printf("\033[48;2;154;205;50m");
            else if (rate >= 55) printf("\033[48;2;255;255;0m");
            else if (rate >= 45) printf("\033[48;2;255;165;0m");
            else if (rate >= 35) printf("\033[48;2;255;99;71m");
            else if (rate >= 25) printf("\033[48;2;220;20;60m");
            else printf("\033[48;2;139;0;0m");
            
            printf(" %2.0f", rate);
            printf("\033[0m");
        }
        printf("║\n");
    }
    
    printf("╚══════╩");
    for (int size = 3; size <= 15; size++)
        printf("═══");
    printf("╝\n");
    
    // 图例
    printf("\n┌─────────────────────────────────────────────────┐\n");
    printf("│ 图例:                                            │\n");
    printf("│ "); 
    printf("\033[48;2;0;150;0m  \033[0m 95-100%%  ");
    printf("\033[48;2;34;139;34m  \033[0m 85-94%%   ");
    printf("\033[48;2;50;205;50m  \033[0m 75-84%%   │\n");
    printf("│ "); 
    printf("\033[48;2;154;205;50m  \033[0m 65-74%%   ");
    printf("\033[48;2;255;255;0m  \033[0m 55-64%%   ");
    printf("\033[48;2;255;165;0m  \033[0m 45-54%%   │\n");
    printf("│ "); 
    printf("\033[48;2;255;99;71m  \033[0m 35-44%%   ");
    printf("\033[48;2;220;20;60m  \033[0m 25-34%%   ");
    printf("\033[48;2;139;0;0m  \033[0m  0-24%%   │\n");
    printf("└─────────────────────────────────────────────────┘\n");
    
    // CSV格式输出
    printf("\n=== CSV格式（可导入Excel） ===\n");
    printf("噪声\\尺寸");
    for (int size = 3; size <= 15; size++)
        printf(",%d", size);
    printf("\n");
    
    for (int n = 0; n < 5; n++) {
        printf("%.0f%%", (n + 1) * 10.0f);
        for (int s = 0; s < 13; s++) {
            printf(",%.1f", results[n][s]);
        }
        printf("\n");
    }
}

// 模式5：全面尺寸测试（3×3到25×25）
void test_all_sizes(int* store_indices, int store_count, float noise_rate) {
    printf("\n╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║              全面尺寸测试 (3×3 到 25×25) - 混合算法                          ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ 存储字母数: %-3d  噪声率: %-5.0f%%  每字母测试: %-3d次                        ║\n", 
           store_count, noise_rate * 100, BATCH_PER_LETTER);
    printf("║ 3×3-15×15: 过采样  |  16×16-25×25: 双线性                                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("%-10s", "尺寸");
    for (int i = 0; i < store_count; i++)
        printf("%-10c", 'A' + store_indices[i]);
    printf("%-10s\n", "总体");
    
    printf("%-10s", "--------");
    for (int i = 0; i < store_count; i++)
        printf("%-10s", "--------");
    printf("%-10s\n", "--------");
    
    for (int size = 3; size <= 25; size++) {
        printf("%-2dx%-2d  ", size, size);
        
        int total_success = 0;
        int total_tests = store_count * BATCH_PER_LETTER;
        
        for (int i = 0; i < store_count; i++) {
            int letter_idx = store_indices[i];
            char letter = 'A' + letter_idx;
            int success_count = 0;
            
            for (int t = 0; t < BATCH_PER_LETTER; t++) {
                int* test_image;
                
                // 根据尺寸选择算法
                if (size <= 15) {
                    // 小尺寸：过采样
                    test_image = generate_test_letter_nearest(letter, size, size, noise_rate);
                    if (!test_image) continue;
                    if (process_input_oversampling(test_image, size, size, letter_idx))
                        success_count++;
                } else {
                    // 大尺寸：双线性
                    test_image = generate_test_letter_bilinear(letter, size, size, noise_rate);
                    if (!test_image) continue;
                    // 双线性标准化
                    int* standardized = standardize_to_5x5_oversampling(test_image, size, size);
                    if (!standardized) {
                        free(test_image);
                        continue;
                    }
                    
                    for (int j = 0; j < net.N; j++)
                        net.S[j] = standardized[j];
                    
                    int* prev = (int*)malloc(net.N * sizeof(int));
                    int steps = 0;
                    
                    while (steps < 50) {
                        for (int j = 0; j < net.N; j++) prev[j] = net.S[j];
                        update_async();
                        steps++;
                        if (is_converged(prev)) break;
                    }
                    
                    int success = 1;
                    for (int j = 0; j < net.N; j++) {
                        if (net.S[j] != base_patterns[letter_idx][j]) {
                            success = 0;
                            break;
                        }
                    }
                    
                    if (success) success_count++;
                    
                    free(standardized);
                    free(prev);
                }
                
                free(test_image);
            }
            
            printf("%-10.1f", success_count * 100.0 / BATCH_PER_LETTER);
            total_success += success_count;
        }
        
        printf("%-10.1f\n", total_success * 100.0 / total_tests);
    }
}

// 模式6：CSV格式输出
void test_csv(int* store_indices, int store_count, float noise_rate) {
    printf("\n=== CSV格式输出（可直接复制到Excel） ===\n\n");
    printf("尺寸");
    for (int i = 0; i < store_count; i++)
        printf(",%c", 'A' + store_indices[i]);
    printf(",总体\n");
    
    for (int size = 3; size <= 25; size++) {
        printf("%d", size);
        int total_success = 0;
        int total_tests = store_count * BATCH_PER_LETTER;
        
        for (int i = 0; i < store_count; i++) {
            int letter_idx = store_indices[i];
            char letter = 'A' + letter_idx;
            int success_count = 0;
            
            for (int t = 0; t < BATCH_PER_LETTER; t++) {
                int* test_image;
                
                if (size <= 15) {
                    test_image = generate_test_letter_nearest(letter, size, size, noise_rate);
                    if (!test_image) continue;
                    if (process_input_oversampling(test_image, size, size, letter_idx))
                        success_count++;
                } else {
                    test_image = generate_test_letter_bilinear(letter, size, size, noise_rate);
                    if (!test_image) continue;
                    
                    int* standardized = standardize_to_5x5_oversampling(test_image, size, size);
                    if (!standardized) {
                        free(test_image);
                        continue;
                    }
                    
                    for (int j = 0; j < net.N; j++)
                        net.S[j] = standardized[j];
                    
                    int* prev = (int*)malloc(net.N * sizeof(int));
                    int steps = 0;
                    
                    while (steps < 50) {
                        for (int j = 0; j < net.N; j++) prev[j] = net.S[j];
                        update_async();
                        steps++;
                        if (is_converged(prev)) break;
                    }
                    
                    int success = 1;
                    for (int j = 0; j < net.N; j++) {
                        if (net.S[j] != base_patterns[letter_idx][j]) {
                            success = 0;
                            break;
                        }
                    }
                    
                    if (success) success_count++;
                    
                    free(standardized);
                    free(prev);
                }
                
                free(test_image);
            }
            
            printf(",%.1f", success_count * 100.0 / BATCH_PER_LETTER);
            total_success += success_count;
        }
        
        printf(",%.1f\n", total_success * 100.0 / total_tests);
    }
    printf("\n提示：复制上述内容，在Excel中使用\"数据→分列\"功能，选择逗号分隔即可。\n");
}

int main() {
    srand((unsigned int)time(NULL));
    
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║     Hopfield网络 - 字母恢复测试系统（混合算法）                  ║\n");
    printf("║     小尺寸(≤15): 过采样  |  大尺寸(>15): 双线性                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    init_network(BASE_N);
    
    // 输入存储的字母
    char input[32];
    printf("请输入要存储的字母（如 ACT 或 ABCDE）: ");
    scanf("%31s", input);
    
    int store_indices[MAX_STORED];
    int store_count = 0;
    
    for (int i = 0; input[i] && store_count < MAX_STORED; i++) {
        char c = input[i];
        if (c >= 'A' && c <= 'Z')
            store_indices[store_count++] = c - 'A';
    }
    
    if (store_count == 0) {
        printf("错误：未输入有效字母。\n");
        free_network();
        return 1;
    }
    
    printf("\n存储字母: ");
    for (int i = 0; i < store_count; i++)
        printf("%c ", 'A' + store_indices[i]);
    printf("\n");
    
    train_network(store_indices, store_count);
    printf("网络训练完成。\n");
    
    // 选择模式
    printf("\n═══════════════════════════════════\n");
    printf("  测试模式选择\n");
    printf("═══════════════════════════════════\n");
    printf("5. 全面尺寸测试 (3×3 到 25×25)\n");
    printf("6. CSV格式输出 (方便导入Excel)\n");
    printf("7. 彩色热力图分析 - 过采样 (3×3 到 15×15)\n");
    printf("═══════════════════════════════════\n");
    printf("请选择 (5-7): ");
    
    int mode;
    scanf("%d", &mode);
    
    if (mode == 5) {
        float noise;
        printf("\n请输入噪声率 (0-1, 如0.3): ");
        scanf("%f", &noise);
        test_all_sizes(store_indices, store_count, noise);
        
        printf("\n是否同时导出CSV格式？(1:是 0:否): ");
        int csv;
        scanf("%d", &csv);
        if (csv == 1) test_csv(store_indices, store_count, noise);
        
    } else if (mode == 6) {
        float noise;
        printf("\n请输入噪声率 (0-1, 如0.3): ");
        scanf("%f", &noise);
        test_csv(store_indices, store_count, noise);
        
    } else if (mode == 7) {
        test_heatmap_oversampling(store_indices, store_count);
        
    } else {
        printf("无效选项。\n");
    }
    
    free_network();
    printf("\n测试完成，感谢使用！\n");
    return 0;
}
