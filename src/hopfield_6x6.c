/*
* Hopfield Network with Dynamic Letter Selection + Batch Test
* 支持：
*   - 存储任意 A-Z 字母子集（6x6 点阵）
*   - 交互式测试单个字母
*   - 批量测试所有已存储字母的成功率
*
* 编译: gcc -o hopfield hopfield_6x6.c -lm
* 运行: ./hopfield
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define N 36                // 6x6 neurons
#define MAX_PATTERNS 26
#define NOISE_RATE 0.3
#define MAX_STORED 15       // 最多允许存15个（防崩溃）
#define BATCH_PER_LETTER 50 // 每个字母测试50次

// ========== 26个字母图案 (A-Z) 6x6 ==========
// 重新设计的6x6图案，每个字母占满6x6方格
int patterns[26][N] = {
    // A
    {-1,-1,+1,+1,-1,-1,
        -1,+1,-1,-1,+1,-1,
        +1,-1,-1,-1,-1,+1,
        +1,+1,+1,+1,+1,+1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1},
    // B
    {+1,+1,+1,+1,-1,-1,
        +1,-1,-1,-1,+1,-1,
        +1,-1,-1,-1,+1,-1,
        +1,+1,+1,+1,-1,-1,
        +1,-1,-1,-1,+1,-1,
        +1,+1,+1,+1,-1,-1},
    // C
    {-1,+1,+1,+1,+1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,-1,-1,-1,-1,-1,
        -1,+1,+1,+1,+1,-1},
    // D
    {+1,+1,+1,+1,-1,-1,
        +1,-1,-1,-1,+1,-1,
        +1,-1,-1,-1,+1,-1,
        +1,-1,-1,-1,+1,-1,
        +1,-1,-1,-1,+1,-1,
        +1,+1,+1,+1,-1,-1},
    // E
    {+1,+1,+1,+1,+1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,+1,+1,+1,-1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,+1,+1,+1,+1,-1},
    // F
    {+1,+1,+1,+1,+1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,+1,+1,+1,-1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,-1,-1,-1,-1,-1},
    // G
    {-1,+1,+1,+1,+1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,-1,-1,+1,+1,-1,
        +1,-1,-1,-1,+1,-1,
        +1,-1,-1,-1,+1,-1,
        -1,+1,+1,+1,+1,-1},
    // H
    {+1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        +1,+1,+1,+1,+1,+1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1},
    // I
    {-1,+1,+1,+1,+1,-1,
        -1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1,
        -1,+1,+1,+1,+1,-1},
    // J
    {-1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,+1,-1,
        -1,-1,-1,-1,+1,-1,
        +1,-1,-1,-1,+1,-1,
        +1,-1,-1,-1,+1,-1,
        -1,+1,+1,+1,-1,-1},
    // K
    {+1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,+1,-1,
        +1,-1,-1,+1,-1,-1,
        +1,+1,+1,-1,-1,-1,
        +1,-1,-1,+1,-1,-1,
        +1,-1,-1,-1,+1,-1},
    // L
    {+1,-1,-1,-1,-1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,+1,+1,+1,+1,-1},
    // M
    {+1,-1,-1,-1,-1,+1,
        +1,+1,-1,-1,+1,+1,
        +1,-1,+1,-1,+1,-1,
        +1,-1,-1,+1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1},
    // N
    {+1,-1,-1,-1,-1,+1,
        +1,+1,-1,-1,-1,+1,
        +1,-1,+1,-1,-1,+1,
        +1,-1,-1,+1,-1,+1,
        +1,-1,-1,-1,+1,+1,
        +1,-1,-1,-1,-1,+1},
    // O
    {-1,+1,+1,+1,+1,-1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        -1,+1,+1,+1,+1,-1},
    // P
    {+1,+1,+1,+1,+1,-1,
        +1,-1,-1,-1,+1,-1,
        +1,-1,-1,-1,+1,-1,
        +1,+1,+1,+1,-1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,-1,-1,-1,-1,-1},
    // Q
    {-1,+1,+1,+1,+1,-1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,-1,+1,-1,+1,
        -1,+1,+1,-1,+1,-1},
    // R
    {+1,+1,+1,+1,+1,-1,
        +1,-1,-1,-1,+1,-1,
        +1,-1,-1,-1,+1,-1,
        +1,+1,+1,+1,-1,-1,
        +1,-1,-1,+1,-1,-1,
        +1,-1,-1,-1,+1,-1},
    // S
    {-1,+1,+1,+1,+1,-1,
        +1,-1,-1,-1,-1,-1,
        +1,-1,-1,-1,-1,-1,
        -1,+1,+1,+1,+1,-1,
        -1,-1,-1,-1,+1,-1,
        +1,+1,+1,+1,-1,-1},
    // T
    {+1,+1,+1,+1,+1,+1,
        -1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1},
    // U
    {+1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        -1,+1,+1,+1,+1,-1},
    // V
    {+1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        -1,+1,-1,-1,+1,-1,
        -1,+1,-1,-1,+1,-1,
        -1,-1,+1,+1,-1,-1,
        -1,-1,+1,+1,-1,-1},
    // W
    {+1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,-1,-1,-1,+1,
        +1,-1,+1,-1,+1,-1,
        +1,+1,-1,+1,+1,-1,
        +1,-1,-1,-1,-1,+1},
    // X
    {+1,-1,-1,-1,-1,+1,
        -1,+1,-1,-1,+1,-1,
        -1,-1,+1,+1,-1,-1,
        -1,-1,+1,+1,-1,-1,
        -1,+1,-1,-1,+1,-1,
        +1,-1,-1,-1,-1,+1},
    // Y
    {+1,-1,-1,-1,-1,+1,
        -1,+1,-1,-1,+1,-1,
        -1,-1,+1,+1,-1,-1,
        -1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1,
        -1,-1,+1,-1,-1,-1},
    // Z
    {+1,+1,+1,+1,+1,+1,
        -1,-1,-1,-1,+1,-1,
        -1,-1,-1,+1,-1,-1,
        -1,-1,+1,-1,-1,-1,
        -1,+1,-1,-1,-1,-1,
        +1,+1,+1,+1,+1,+1}
};

// ========== 全局变量 ==========
double J[N][N];
int S[N];

// ========== 工具函数 ==========
void print_state() {
    for (int i = 0; i < N; i++) {
        if (i % 6 == 0) printf("\n");
        if (S[i] == +1) printf("█");
        else            printf(" ");
    }
    printf("\n\n");
}

void train_from_list(int* indices, int count) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            J[i][j] = 0.0;
    
    for (int mu = 0; mu < count; mu++) {
        int idx = indices[mu];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    J[i][j] += (double)(patterns[idx][i] * patterns[idx][j]);
                }
            }
        }
    }
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            J[i][j] /= N;
}

void add_noise(int pattern_idx) {
    for (int i = 0; i < N; i++) {
        S[i] = patterns[pattern_idx][i];
        if ((double)rand() / RAND_MAX < NOISE_RATE) {
            S[i] = -S[i];
        }
    }
}

void update_async() {
    int order[N];
    for (int i = 0; i < N; i++) order[i] = i;
    for (int i = N-1; i > 0; i--) {
        int j = rand() % (i+1);
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }
    for (int k = 0; k < N; k++) {
        int i = order[k];
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) sum += J[i][j] * S[j];
        }
        S[i] = (sum >= 0) ? +1 : -1;
    }
}

int is_converged(int prev[N]) {
    for (int i = 0; i < N; i++)
        if (S[i] != prev[i]) return 0;
    return 1;
}

char get_letter(int idx) { return 'A' + idx; }

// 单次无打印测试
int run_single_test(int pattern_idx) {
    add_noise(pattern_idx);
    int prev[N], steps = 0;
    while (steps < 20) {
        for (int i = 0; i < N; i++) prev[i] = S[i];
        update_async();
        steps++;
        if (is_converged(prev)) break;
    }
    for (int i = 0; i < N; i++) {
        if (S[i] != patterns[pattern_idx][i]) return 0;
    }
    return 1;
}

// 交互式测试
void test_interactive(int pattern_idx) {
    char c = get_letter(pattern_idx);
    printf("原始记忆 %c:\n", c);
    for (int i = 0; i < N; i++) S[i] = patterns[pattern_idx][i];
    print_state();
    
    add_noise(pattern_idx);
    printf("输入（带 %.0f%% 噪声）:\n", NOISE_RATE * 100);
    print_state();
    
    int prev[N], steps = 0;
    while (steps < 20) {
        for (int i = 0; i < N; i++) prev[i] = S[i];
        update_async();
        steps++;
        if (is_converged(prev)) break;
    }
    
    printf("网络输出（经 %d 步更新）:\n", steps);
    print_state();
    
    int success = 1;
    for (int i = 0; i < N; i++) {
        if (S[i] != patterns[pattern_idx][i]) {
            success = 0;
            break;
        }
    }
    printf("恢复结果: %s\n", success ? "成功!" : "失败!");
}

// 批量测试
void batch_test(int* store_indices, int store_count) {
    printf("\n>>> 开始批量测试（每个字母 %d 次）...\n", BATCH_PER_LETTER);
    
    int total_success = 0;
    int success_per_letter[MAX_STORED] = {0};
    
    for (int i = 0; i < store_count; i++) {
        int idx = store_indices[i];
        int success = 0;
        for (int t = 0; t < BATCH_PER_LETTER; t++) {
            if (run_single_test(idx)) success++;
        }
        success_per_letter[i] = success;
        total_success += success;
    }
    
    printf("\n=== 批量测试结果 ===\n");
    for (int i = 0; i < store_count; i++) {
        char c = get_letter(store_indices[i]);
        double rate = success_per_letter[i] * 100.0 / BATCH_PER_LETTER;
        printf("字母 %c: %d / %d 成功 (%.1f%%)\n", c, success_per_letter[i], BATCH_PER_LETTER, rate);
    }
    double overall = total_success * 100.0 / (store_count * BATCH_PER_LETTER);
    printf("总体成功率: %d / %d (%.1f%%)\n", total_success, store_count * BATCH_PER_LETTER, overall);
    
    // 容量提示
    double theoretical_capacity = 0.14 * N; // ≈5.04
    if (store_count > theoretical_capacity) {
        printf("\n?? 提示：存储数量 (%d) 超过理论容量 (%.1f)，成功率下降是正常现象。\n", store_count, theoretical_capacity);
    }
}

// ========== 主函数 ==========
int main() {
    srand(time(NULL));
    
    printf("=== 动态 Hopfield 联想记忆 (A-Z, 6x6) ===\n");
    printf("支持存储任意 A-Z 字母子集，并测试恢复能力\n\n");
    
    // 1. 输入要存储的字母
    char input[32];
    printf("请输入要存储的字母（如 ACT 或 ABCDE）: ");
    scanf("%31s", input);
    
    int store_indices[MAX_STORED];
    int store_count = 0;
    for (int i = 0; input[i] && store_count < MAX_STORED; i++) {
        char c = input[i];
        if (c >= 'A' && c <= 'Z') {
            store_indices[store_count++] = c - 'A';
        }
    }
    
    if (store_count == 0) {
        printf("错误：未输入有效字母。\n");
        return 1;
    }
    
    printf("将存储以下 %d 个字母: ", store_count);
    for (int i = 0; i < store_count; i++) {
        printf("%c ", get_letter(store_indices[i]));
    }
    printf("\n\n");
    
    // 2. 训练网络
    train_from_list(store_indices, store_count);
    
    // 3. 选择模式
    printf("请选择操作:\n");
    printf("  1. 交互式测试（可视化过程）\n");
    printf("  2. 批量测试（统计成功率）\n");
    printf("请输入选项 (1 或 2): ");
    
    int mode;
    if (scanf("%d", &mode) != 1) mode = 2;
    
    if (mode == 1) {
        // 交互式测试
        printf("请选择要测试的字母（必须在上述列表中）: ");
        char test_char;
        scanf(" %c", &test_char);
        if (test_char < 'A' || test_char > 'Z') {
            printf("无效字母！\n");
            return 1;
        }
        int test_idx = test_char - 'A';
        
        int found = 0;
        for (int i = 0; i < store_count; i++) {
            if (store_indices[i] == test_idx) {
                found = 1;
                break;
            }
        }
        if (!found) {
            printf("错误：%c 未被存储！\n", test_char);
            return 1;
        }
        test_interactive(test_idx);
    } else {
        // 批量测试
        batch_test(store_indices, store_count);
    }
    
    return 0;
}

