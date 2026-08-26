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

typedef struct {
    int N;
    double** J;
    int* S;
} DynamicNetwork;

DynamicNetwork net;
int current_N = BASE_N;

int* generate_test_letter(char letter, int width, int height, float noise_rate) {
    if (width <= 0 || height <= 0) return NULL;
    int* image = (int*)malloc(width * height * sizeof(int));
    if (!image) return NULL;
    int base_idx = letter - 'A';
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int src_x = (x * 5) / width;
            int src_y = (y * 5) / height;
            if (src_x >= 5) src_x = 4;
            if (src_y >= 5) src_y = 4;
            int pixel_val = base_patterns[base_idx][src_y * 5 + src_x];
            if ((float)rand() / RAND_MAX < noise_rate) pixel_val = -pixel_val;
            image[y * width + x] = pixel_val;
        }
    }
    return image;
}

int* standardize_to_5x5_oversampling(int* input, int width, int height) {
    int* output = (int*)malloc(25 * sizeof(int));
    if (!output) return NULL;
    int expand_factor = 5;
    int expanded_w = width * expand_factor;
    int expanded_h = height * expand_factor;
    int* expanded = (int*)malloc(expanded_w * expanded_h * sizeof(int));
    if (!expanded) { free(output); return NULL; }
    
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            int value = input[y * width + x];
            for (int dy = 0; dy < expand_factor; dy++)
                for (int dx = 0; dx < expand_factor; dx++)
                    expanded[(y * expand_factor + dy) * expanded_w + (x * expand_factor + dx)] = value;
        }
    
    int cell_w = expanded_w / 5;
    int cell_h = expanded_h / 5;
    for (int ty = 0; ty < 5; ty++)
        for (int tx = 0; tx < 5; tx++) {
            int plus_count = 0, total = 0;
            for (int dy = 0; dy < cell_h; dy++)
                for (int dx = 0; dx < cell_w; dx++) {
                    int idx = (ty * cell_h + dy) * expanded_w + (tx * cell_w + dx);
                    if (expanded[idx] == +1) plus_count++;
                    total++;
                }
            output[ty * 5 + tx] = (plus_count > total / 2) ? +1 : -1;
        }
    free(expanded);
    return output;
}

void init_network(int N) {
    current_N = N;
    net.N = N;
    net.J = (double**)malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) net.J[i] = (double*)calloc(N, sizeof(double));
    net.S = (int*)malloc(N * sizeof(int));
}

void free_network() {
    if (net.J) {
        for (int i = 0; i < current_N; i++) free(net.J[i]);
        free(net.J);
    }
    if (net.S) free(net.S);
}

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

// 标准异步更新
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

// 带退火的异步更新
void update_async_annealing(float temperature) {
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
        double prob_plus = 1.0 / (1.0 + exp(-2.0 * sum / temperature));
        net.S[i] = ((double)rand() / RAND_MAX < prob_plus) ? +1 : -1;
    }
    free(order);
}

int is_converged(int prev[]) {
    for (int i = 0; i < net.N; i++)
        if (net.S[i] != prev[i]) return 0;
    return 1;
}

// 检查当前状态是否等于某个字母
int check_if_pattern(int* state) {
    for (int p = 0; p < 26; p++) {
        int match = 1;
        for (int i = 0; i < 25; i++) {
            if (state[i] != base_patterns[p][i]) {
                match = 0;
                break;
            }
        }
        if (match) return p;
    }
    return -1;
}

// ========== 核心：只在陷入非字母局部极小时才退火 ==========
int process_input(int* input_pattern, int width, int height, int target_idx) {
    int* standardized = standardize_to_5x5_oversampling(input_pattern, width, height);
    if (!standardized) return 0;
    
    // 第一步：标准恢复
    for (int i = 0; i < net.N; i++)
        net.S[i] = standardized[i];
    
    int* prev = (int*)malloc(net.N * sizeof(int));
    int steps = 0;
    while (steps < 30) {
        for (int i = 0; i < net.N; i++) prev[i] = net.S[i];
        update_async();
        steps++;
        if (is_converged(prev)) break;
    }
    
    // 检查收敛到了什么
    int result = check_if_pattern(net.S);
    
    if (result >= 0) {
        // 收敛到了某个字母，直接返回结果
        int success = (result == target_idx) ? 1 : 0;
        free(standardized);
        free(prev);
        return success;
    }
    
    // 第二步：陷入了非字母的局部极小值！启动退火
    // 重新初始化状态
    for (int i = 0; i < net.N; i++)
        net.S[i] = standardized[i];
    
    float T = 3.0;
    while (T > 0.1) {
        update_async_annealing(T);
        T *= 0.85;
        
        // 检查是否跳到了某个字母
        result = check_if_pattern(net.S);
        if (result >= 0) {
            int success = (result == target_idx) ? 1 : 0;
            free(standardized);
            free(prev);
            return success;
        }
    }
    
    // 退火结束，低温收敛
    steps = 0;
    while (steps < 20) {
        for (int i = 0; i < net.N; i++) prev[i] = net.S[i];
        update_async();
        steps++;
        if (is_converged(prev)) break;
    }
    
    // 最终检查
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

// 模式7：热力图
void test_heatmap(int* store_indices, int store_count) {
    printf("\n╔══════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║         热力图分析 - 过采样+标准恢复（陷入非字母极小时启用退火）                         ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ 存储字母数: %-3d  每组合测试: %-3d次/字母                                                 ║\n", 
           store_count, BATCH_PER_LETTER);
    printf("║ 策略: 标准恢复 → 若收敛到字母则结束 → 若陷入非字母极小则退火                             ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n正在测试，请稍候...\n\n");
    
    float results[5][23];
    
    for (int n = 0; n < 5; n++) {
        float noise_rate = (n + 1) * 0.1f;
        for (int s = 0; s < 23; s++) {
            int size = s + 3;
            int total_success = 0;
            int total_tests = store_count * BATCH_PER_LETTER;
            for (int i = 0; i < store_count; i++) {
                int letter_idx = store_indices[i];
                char letter = 'A' + letter_idx;
                for (int t = 0; t < BATCH_PER_LETTER; t++) {
                    int* test_image = generate_test_letter(letter, size, size, noise_rate);
                    if (!test_image) continue;
                    if (process_input(test_image, size, size, letter_idx))
                        total_success++;
                    free(test_image);
                }
            }
            results[n][s] = total_success * 100.0f / total_tests;
        }
        printf("  噪声 %.0f%% 测试完成\n", noise_rate * 100);
    }
    
    printf("\n");
    printf("╔══════╦");
    for (int size = 3; size <= 25; size++) printf("═══");
    printf("══╗\n");
    printf("║噪声%% ║");
    for (int size = 3; size <= 25; size++) printf("%3d", size);
    printf(" ║\n");
    printf("╠══════╬");
    for (int size = 3; size <= 25; size++) printf("═══");
    printf("══╣\n");
    
    for (int n = 0; n < 5; n++) {
        printf("║ %3.0f%% ║", (n + 1) * 10.0f);
        for (int s = 0; s < 23; s++) {
            float rate = results[n][s];
            if (rate >= 95) printf("\033[48;2;0;150;0m");
            else if (rate >= 85) printf("\033[48;2;34;139;34m");
            else if (rate >= 75) printf("\033[48;2;50;205;50m");
            else if (rate >= 65) printf("\033[48;2;154;205;50m");
            else if (rate >= 55) printf("\033[48;2;255;255;0m");
            else if (rate >= 45) printf("\033[48;2;255;165;0m");
            else if (rate >= 35) printf("\033[48;2;255;99;71m");
            else if (rate >= 25) printf("\033[48;2;220;20;60m");
            else printf("\033[48;2;139;0;0m");
            printf("%3.0f", rate);
            printf("\033[0m");
        }
        printf(" ║\n");
    }
    printf("╚══════╩");
    for (int size = 3; size <= 25; size++) printf("═══");
    printf("══╝\n");
    
    printf("\n┌─────────────────────────────────────────────────┐\n");
    printf("│ 图例:                                            │\n");
    printf("│ "); 
    printf("\033[48;2;0;150;0m   \033[0m 95-100%%  ");
    printf("\033[48;2;34;139;34m   \033[0m 85-94%%   ");
    printf("\033[48;2;50;205;50m   \033[0m 75-84%%   │\n");
    printf("│ "); 
    printf("\033[48;2;154;205;50m   \033[0m 65-74%%   ");
    printf("\033[48;2;255;255;0m   \033[0m 55-64%%   ");
    printf("\033[48;2;255;165;0m   \033[0m 45-54%%   │\n");
    printf("│ "); 
    printf("\033[48;2;255;99;71m   \033[0m 35-44%%   ");
    printf("\033[48;2;220;20;60m   \033[0m 25-34%%   ");
    printf("\033[48;2;139;0;0m   \033[0m  0-24%%   │\n");
    printf("└─────────────────────────────────────────────────┘\n");
    
    printf("\n=== CSV格式 ===\n");
    printf("噪声\\尺寸");
    for (int size = 3; size <= 25; size++) printf(",%d", size);
    printf("\n");
    for (int n = 0; n < 5; n++) {
        printf("%.0f%%", (n + 1) * 10.0f);
        for (int s = 0; s < 23; s++) printf(",%.1f", results[n][s]);
        printf("\n");
    }
}

int main() {
    srand((unsigned int)time(NULL));
    
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║   Hopfield网络 - 过采样+退火（仅非字母极小值启用）              ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    init_network(BASE_N);
    
    char input[32];
    printf("请输入要存储的字母（如 ACT 或 ABCDE）: ");
    scanf("%31s", input);
    
    int store_indices[MAX_STORED];
    int store_count = 0;
    for (int i = 0; input[i] && store_count < MAX_STORED; i++) {
        char c = input[i];
        if (c >= 'A' && c <= 'Z') store_indices[store_count++] = c - 'A';
    }
    
    if (store_count == 0) {
        printf("错误：未输入有效字母。\n");
        free_network();
        return 1;
    }
    
    printf("\n存储字母: ");
    for (int i = 0; i < store_count; i++) printf("%c ", 'A' + store_indices[i]);
    printf("\n");
    train_network(store_indices, store_count);
    printf("网络训练完成。\n开始热力图测试...\n");
    
    test_heatmap(store_indices, store_count);
    
    free_network();
    printf("\n测试完成！\n");
    return 0;
}
