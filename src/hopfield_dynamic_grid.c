#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define BASE_N 25 // 基础神经元数量（5×5 网格）
#define MAX_PATTERNS 26 //最多26个字母
#define NOISE_RATE 0.3 //30% 噪声
#define MAX_STORED 15 // 最多允许存15个（防崩溃）
#define BATCH_PER_LETTER 50 // 批量测试：每个字母测试50次

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
    int N;          // 当前网络的神经元数量
    double** J;     // 动态权重矩阵（N×N）
    int* S;         // 当前神经元状态
} DynamicNetwork;

// 全局变量
DynamicNetwork net;
int current_N = BASE_N; // 当前使用的神经元数

// ========== 修正的双线性插值算法（保持对称性）==========

// 生成指定尺寸的字母测试图像（使用对称双线性插值）
int* generate_test_letter_symmetric(char letter, int width, int height, float noise_rate) {
    // 防止无效尺寸
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "错误: 图像尺寸无效 (%dx%d)\n", width, height);
        return NULL;
    }
    
    int total_pixels = width * height;
    // 检查整数溢出
    if (total_pixels / width != height) {
        fprintf(stderr, "错误: 图像尺寸过大导致溢出\n");
        return NULL;
    }
    
    int* image = (int*)malloc(total_pixels * sizeof(int));
    if (!image) {
        fprintf(stderr, "错误: 内存分配失败 (%d 像素)\n", total_pixels);
        return NULL;
    }
    
    int base_idx = letter - 'A';
    
    // ========== 关键修正：使用对称映射 ==========
    // 中心点和缩放因子（对称映射）
    float center_x = (width - 1) / 2.0f;
    float center_y = (height - 1) / 2.0f;
    float scale_x = 4.0f / (width - 1);  // (5-1)/(N-1)
    float scale_y = 4.0f / (height - 1);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 对称映射到5×5的浮点坐标
            float src_x = (x - center_x) * scale_x + 2.0f;  // +2是5×5中心索引2
            float src_y = (y - center_y) * scale_y + 2.0f;
            
            // 边界处理
            if (src_x < 0.0f) src_x = 0.0f;
            if (src_x > 4.0f) src_x = 4.0f;
            if (src_y < 0.0f) src_y = 0.0f;
            if (src_y > 4.0f) src_y = 4.0f;
            
            // 双线性插值
            int x1 = (int)src_x;
            int y1 = (int)src_y;
            int x2 = x1 + 1;
            int y2 = y1 + 1;
            
            // 边界检查
            if (x2 > 4) x2 = 4;
            if (y2 > 4) y2 = 4;
            
            float wx = src_x - x1;
            float wy = src_y - y1;
            
            // 获取四个点的值（转换为0/1）
            float v11 = (base_patterns[base_idx][y1*5 + x1] == +1) ? 1.0f : 0.0f;
            float v12 = (base_patterns[base_idx][y1*5 + x2] == +1) ? 1.0f : 0.0f;
            float v21 = (base_patterns[base_idx][y2*5 + x1] == +1) ? 1.0f : 0.0f;
            float v22 = (base_patterns[base_idx][y2*5 + x2] == +1) ? 1.0f : 0.0f;
            
            // 双线性插值
            float interpolated = 
                v11 * (1-wx) * (1-wy) +
                v12 * wx * (1-wy) +
                v21 * (1-wx) * wy +
                v22 * wx * wy;
            
            // 二值化
            int pixel_val = (interpolated > 0.5f) ? +1 : -1;
            
            // 添加噪声
            if ((float)rand() / RAND_MAX < noise_rate) {
                pixel_val = -pixel_val;
            }
            
            image[y * width + x] = pixel_val;
        }
    }
    
    return image;
}

// ========== 标准化算法 ==========

// 将任意尺寸图像标准化到5×5（使用对称感知的标准化）
int* standardize_to_5x5_symmetric(int* input, int width, int height) {
    int* output = (int*)malloc(25 * sizeof(int));
    if (!output) return NULL;
    
    // 使用对称感知的标准化
    for (int ty = 0; ty < 5; ty++) {
        for (int tx = 0; tx < 5; tx++) {
            // 对称映射：将5×5网格单元映射回输入图像
            float center_x = (width - 1) / 2.0f;
            float center_y = (height - 1) / 2.0f;
            float scale_x = (width - 1) / 4.0f;  // 反向映射
            float scale_y = (height - 1) / 4.0f;
            
            // 5×5网格位置映射到输入图像
            float src_x = (tx - 2.0f) * scale_x + center_x;
            float src_y = (ty - 2.0f) * scale_y + center_y;
            
            // 边界处理
            if (src_x < 0) src_x = 0;
            if (src_x >= width) src_x = width - 1;
            if (src_y < 0) src_y = 0;
            if (src_y >= height) src_y = height - 1;
            
            // 取最近的像素值
            int nearest_x = (int)(src_x + 0.5f);
            int nearest_y = (int)(src_y + 0.5f);
            
            // 确保在范围内
            if (nearest_x < 0) nearest_x = 0;
            if (nearest_x >= width) nearest_x = width - 1;
            if (nearest_y < 0) nearest_y = 0;
            if (nearest_y >= height) nearest_y = height - 1;
            
            output[ty * 5 + tx] = input[nearest_y * width + nearest_x];
        }
    }
    
    return output;
}

// 打印任意尺寸的图像
void print_image(int* image, int width, int height, const char* label) {
    printf("%s (%dx%d):\n", label, width, height);
    
    // 打印顶部边框
    printf("┌");
    for (int x = 0; x < width; x++) printf("─");
    printf("┐\n");
    
    for (int y = 0; y < height; y++) {
        printf("│");
        for (int x = 0; x < width; x++) {
            printf("%s", image[y * width + x] == +1 ? "█" : " ");
        }
        printf("│\n");
    }
    
    // 打印底部边框
    printf("└");
    for (int x = 0; x < width; x++) printf("─");
    printf("┘\n\n");
}

// ========== 动态网络函数 ==========

// 初始化动态网络
void init_dynamic_network(int N) {
    current_N = N;
    net.N = N;
    
    // 分配权重矩阵
    net.J = (double**)malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        net.J[i] = (double*)calloc(N, sizeof(double));
    }
    
    // 分配状态向量
    net.S = (int*)malloc(N * sizeof(int));
}

// 释放动态网络内存
void free_dynamic_network() {
    if (net.J) {
        for (int i = 0; i < current_N; i++) {
            free(net.J[i]);
        }
        free(net.J);
    }
    if (net.S) free(net.S);
}

// 训练网络（存储字母）
void train_from_list(int* indices, int count) {
    int N = net.N;
    
    // 清零权重矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            net.J[i][j] = 0.0;
        }
    }
    
    // Hebb学习规则
    for (int mu = 0; mu < count; mu++) {
        int idx = indices[mu];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    net.J[i][j] += (double)(base_patterns[idx][i] * base_patterns[idx][j]);
                }
            }
        }
    }
    
    // 归一化
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            net.J[i][j] /= N;
        }
    }
}

// 异步更新
void update_async() {
    int N = net.N;
    int* order = (int*)malloc(N * sizeof(int));
    
    // 创建随机更新顺序
    for (int i = 0; i < N; i++) order[i] = i;
    for (int i = N-1; i > 0; i--) {
        int j = rand() % (i+1);
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }
    
    // 按顺序更新每个神经元
    for (int k = 0; k < N; k++) {
        int i = order[k];
        double sum = 0.0;
        
        for (int j = 0; j < N; j++) {
            if (i != j) sum += net.J[i][j] * net.S[j];
        }
        
        net.S[i] = (sum >= 0) ? +1 : -1;
    }
    
    free(order);
}

// 检查是否收敛
int is_converged(int prev[]) {
    for (int i = 0; i < net.N; i++) {
        if (net.S[i] != prev[i]) return 0;
    }
    return 1;
}

// 单次测试（无打印）
int run_single_test(int pattern_idx) {
    // 设置初始状态为目标图案
    for (int i = 0; i < net.N; i++) {
        net.S[i] = base_patterns[pattern_idx][i];
    }
    
    // 添加噪声
    for (int i = 0; i < net.N; i++) {
        if ((double)rand() / RAND_MAX < NOISE_RATE) {
            net.S[i] = -net.S[i];
        }
    }
    
    // 迭代更新直到收敛
    int* prev = (int*)malloc(net.N * sizeof(int));
    int steps = 0;
    
    while (steps < 50) { // 增加最大步数
        for (int i = 0; i < net.N; i++) prev[i] = net.S[i];
        update_async();
        steps++;
        if (is_converged(prev)) break;
    }
    
    // 检查是否恢复成功
    int success = 1;
    for (int i = 0; i < net.N; i++) {
        if (net.S[i] != base_patterns[pattern_idx][i]) {
            success = 0;
            break;
        }
    }
    
    free(prev);
    return success;
}

// 处理任意输入并恢复（使用对称标准化）
int process_arbitrary_input_symmetric(int* input_pattern, int width, int height, int target_letter_idx) {
    // 1. 使用对称感知标准化到5×5
    int* standardized = standardize_to_5x5_symmetric(input_pattern, width, height);
    if (!standardized) return 0;
    
    // 2. 设置网络初始状态
    for (int i = 0; i < net.N; i++) {
        net.S[i] = standardized[i];
    }
    
    // 3. 迭代更新
    int* prev = (int*)malloc(net.N * sizeof(int));
    int steps = 0;
    
    while (steps < 50) {
        for (int i = 0; i < net.N; i++) prev[i] = net.S[i];
        update_async();
        steps++;
        if (is_converged(prev)) break;
    }
    
    // 4. 检查是否恢复为目标字母
    int success = 1;
    for (int i = 0; i < net.N; i++) {
        if (net.S[i] != base_patterns[target_letter_idx][i]) {
            success = 0;
            break;
        }
    }
    
    free(standardized);
    free(prev);
    return success;
}

// ========== 批量测试函数 ==========

void batch_test_arbitrary_input_symmetric(int* store_indices, int store_count, 
                                          int test_width, int test_height, float noise_rate) {
    printf("\n>>> 开始批量测试（每个字母50次，输入尺寸：%dx%d，噪声：%.0f%%）...\n", 
           test_width, test_height, noise_rate * 100);
    
    int total_success = 0;
    int total_tests = store_count * 50;
    
    for (int i = 0; i < store_count; i++) {
        int letter_idx = store_indices[i];
        char letter = 'A' + letter_idx;
        int success_count = 0;
        
        for (int t = 0; t < 50; t++) {
            // 生成测试图像（使用对称算法）
            int* test_image = generate_test_letter_symmetric(letter, test_width, test_height, noise_rate);
            if (!test_image) continue;
            
            // 处理并测试（使用对称标准化）
            if (process_arbitrary_input_symmetric(test_image, test_width, test_height, letter_idx)) {
                success_count++;
            }
            
            free(test_image);
        }
        
        double rate = success_count * 100.0 / 50;
        printf("字母 %c: %d/50 成功 (%.1f%%)\n", letter, success_count, rate);
        total_success += success_count;
    }
    
    double overall_rate = total_success * 100.0 / total_tests;
    printf("\n总体成功率: %d/%d (%.1f%%)\n", total_success, total_tests, overall_rate);
}

// ========== 演示函数 ==========

void demo_single_case_symmetric(char letter, int width, int height, float noise_rate) {
    int letter_idx = letter - 'A';
    
    printf("\n=== 演示案例（对称算法） ===\n");
    printf("目标字母: %c\n", letter);
    printf("输入尺寸: %dx%d\n", width, height);
    printf("噪声水平: %.0f%%\n", noise_rate * 100);
    
    // 1. 生成测试图像（使用对称算法）
    int* test_image = generate_test_letter_symmetric(letter, width, height, noise_rate);
    if (!test_image) {
        printf("错误：生成图像失败\n");
        return;
    }
    print_image(test_image, width, height, "生成的测试图像（对称算法）");
    
    // 2. 标准化到5×5（使用对称标准化）
    int* standardized = standardize_to_5x5_symmetric(test_image, width, height);
    if (!standardized) {
        free(test_image);
        printf("错误：标准化失败\n");
        return;
    }
    print_image(standardized, 5, 5, "标准化后的5×5图像");
    
    // 3. 处理并恢复
    printf("开始网络恢复...\n");
    for (int i = 0; i < net.N; i++) {
        net.S[i] = standardized[i];
    }
    
    int* prev = (int*)malloc(net.N * sizeof(int));
    int steps = 0;
    
    while (steps < 50) {
        for (int i = 0; i < net.N; i++) prev[i] = net.S[i];
        update_async();
        steps++;
        if (is_converged(prev)) break;
    }
    
    print_image(net.S, 5, 5, "网络恢复结果");
    
    // 4. 检查结果
    int success = 1;
    for (int i = 0; i < net.N; i++) {
        if (net.S[i] != base_patterns[letter_idx][i]) {
            success = 0;
            break;
        }
    }
    
    printf("恢复结果: %s\n", success ? "成功!" : "失败!");
    printf("迭代步数: %d\n", steps);
    
    free(test_image);
    free(standardized);
    free(prev);
}

// ========== 主函数 ==========

int main() {
    srand((unsigned int)time(NULL));
    
    printf("=== 动态Hopfield网络 - 任意输入字母恢复（对称算法） ===\n");
    printf("使用对称双线性插值算法，保持图像对称性\n");
    
    // 1. 初始化网络（固定为5×5）
    init_dynamic_network(BASE_N);
    
    // 2. 输入要存储的字母
    char input[32];
    printf("\n请输入要存储的字母（如 ACT 或 ABCDE）: ");
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
        free_dynamic_network();
        return 1;
    }
    
    printf("将存储以下 %d 个字母: ", store_count);
    for (int i = 0; i < store_count; i++) {
        printf("%c ", 'A' + store_indices[i]);
    }
    printf("\n\n");
    
    // 3. 训练网络
    train_from_list(store_indices, store_count);
    printf("网络训练完成。\n");
    
    // 4. 选择测试模式（已移除对称性测试选项）
    printf("\n请选择测试模式:\n");
    printf("1. 单案例演示（可视化过程）\n");
    printf("2. 批量测试不同尺寸\n");
    printf("3. 批量测试不同噪声水平\n");
    printf("4. 综合测试\n");
    printf("请输入选项 (1-4): ");
    
    int mode;
    scanf("%d", &mode);
    
    if (mode == 1) {
        // 单案例演示
        char test_letter;
        printf("请输入测试字母: ");
        scanf(" %c", &test_letter);
        
        if (test_letter < 'A' || test_letter > 'Z') {
            printf("无效字母！\n");
            free_dynamic_network();
            return 1;
        }
        
        int width, height;
        float noise;
        
        printf("请输入图像宽度 (建议≥5): ");
        scanf("%d", &width);
        printf("请输入图像高度 (建议≥5): ");
        scanf("%d", &height);
        printf("请输入噪声率 (0-1, 如0.3): ");
        scanf("%f", &noise);
        
        // 确保尺寸至少为1
        if (width < 1) width = 1;
        if (height < 1) height = 1;
        
        demo_single_case_symmetric(test_letter, width, height, noise);
        
    } else if (mode == 2) {
        // 批量测试不同尺寸
        printf("\n=== 测试不同输入尺寸的影响 ===\n");
        
        int test_sizes[][2] = {{8,8}, {10,10}, {15,15}, {20,20}, {25,25}};
        int num_sizes = 5;
        
        for (int s = 0; s < num_sizes; s++) {
            int w = test_sizes[s][0];
            int h = test_sizes[s][1];
            
            printf("\n--- 测试尺寸: %dx%d ---\n", w, h);
            batch_test_arbitrary_input_symmetric(store_indices, store_count, w, h, 0.3);
        }
        
    } else if (mode == 3) {
        // 批量测试不同噪声水平
        printf("\n=== 测试不同噪声水平的影响 ===\n");
        
        float noise_levels[] = {0.1, 0.2, 0.3, 0.4, 0.5};
        int num_levels = 5;
        
        for (int n = 0; n < num_levels; n++) {
            printf("\n--- 测试噪声: %.0f%% ---\n", noise_levels[n] * 100);
            batch_test_arbitrary_input_symmetric(store_indices, store_count, 15, 15, noise_levels[n]);
        }
        
    } else if (mode == 4) {
        // 综合测试
        printf("\n=== 综合测试 ===\n");
        
        // 测试1：基础尺寸，基础噪声
        printf("\n1. 基础测试 (10x10, 30%%噪声):\n");
        batch_test_arbitrary_input_symmetric(store_indices, store_count, 10, 10, 0.3);
        
        // 测试2：大尺寸，低噪声
        printf("\n2. 大尺寸低噪声测试 (20x20, 10%%噪声):\n");
        batch_test_arbitrary_input_symmetric(store_indices, store_count, 20, 20, 0.1);
        
        // 测试3：小尺寸，高噪声
        printf("\n3. 小尺寸高噪声测试 (8x8, 50%%噪声):\n");
        batch_test_arbitrary_input_symmetric(store_indices, store_count, 8, 8, 0.5);
        
        // 测试4：7×7对称测试（已删除，该部分不再存在）
    } else {
        printf("无效选项。\n");
    }
    
    // 5. 清理内存
    free_dynamic_network();
    printf("\n测试完成。\n");
    
    return 0;
}
