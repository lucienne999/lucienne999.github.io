---
title: "动态规划"
category: "代码"
tags: ["DP", "LeetCode"]
date: 2026-01-12
---

# 什么是动态规划 (DP)

DP 的核心在于两个特征：**最优子结构** 和 **重叠子问题**。

以经典的 **Fibonacci 数列** 为例：
*   **最优子结构**：`F(n)` 由 `F(n−1)` 和 `F(n−2)` 组合而成，即大问题的最优解包含小问题的最优解。
*   **重叠子问题**：计算 `F(6)` 和 `F(5)` 时，都需要计算 `F(4)`。DP 的精髓就是“记账”，算过的就不再算。

**与其他算法的区别：**
*   **贪心**：只选当下最优，不考虑子问题重叠（不回头）。
*   **二分**：通过排除法缩小范围，不涉及子结构组合（不拼凑）。

所以，选武器前，先看清碉堡的构造。

# DP 的两种形式

1. **Top-Down 自顶向下**

    实现方式 ： **递归 + 备忘录**。
    
    为了防止重复计算（比如算 $F(19)$ 和 $F(18)$ 时都会用到 $F(17)$），我们把算过的结果存到一个表（Memo）里。下次遇到直接查表。

    ```python
    memo = {}
    def fib(n):
        if n == 0 or n == 1: return n
        if n in memo: return memo[n]  # 查表
        
        # 没算过？算一遍并存起来
        memo[n] = fib(n-1) + fib(n-2)
        return memo[n]
    ```

2. **Bottom-Up 自底向上**

    实现方式 ： **循环 (迭代) + DP Table**。
    
    通常不需要递归，直接用 `for` 循环从最小的状态开始填数组。

    ```python
    def fib(n):
        if n == 0 or n == 1: return n
        dp = [0] * (n + 1)
        dp[0], dp[1] = 0, 1
        
        # 从底向上填表
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
            
        return dp[n]
    ```


虽然两者的时间复杂度通常一样（都是 $O(N)$），但在工程实践中：

1.  **函数调用开销**：
    *   Top-down 依赖递归，每次函数调用都要压栈、保存现场、出栈，这在深度很大时是一笔不小的开销。
    *   Bottom-up 只是简单的 `for` 循环，指令更紧凑，CPU 预测更准。

2.  **“偷懒”的 Top-down**：
    *   Top-down 有个独特的优势：**按需计算 (Lazy Evaluation)**。
    *   如果某些子问题对于最终答案根本不需要，Top-down 就不会去算它。
    *   而 Bottom-up 通常会把整个表格填满。如果表格很大但实际用到的格子很少，Top-down 反而会更快。


# 什么时候该用 DP？

如果题目问以下三类问题，大概率是 DP：

1.  **求最值**：最大利润、最长子序列、最小花费。
2.  **求可行性**：能不能拼出某个数、能不能到达终点。
3.  **求方案数**：有多少种走法、有多少种组合。

**核心判断标准**：
如果题目要求你做决策（选或不选），且现在的选择会影响未来的结果，但不需要知道“具体是怎么选的”（只关心结果），大概率是 DP。


# DP 解题五部曲 (The Strategy)

1. 定义 DP 数组含义 (State Definition)

核心问题：dp[i] 或 dp[i][j] 到底代表什么？

Tip：通常是题目要求的值（如：最大利润、最长长度、方案数）。Example：在“爬楼梯”中，dp[i] 表示“走到第 $i$ 阶的方法数”。

2. 确定递推公式 (State Transition)

核心问题：dp[i] 是怎么由之前的状态（dp[i-1], dp[i-2], ...）推导出来的？

Tip：这就是“找关系”。也就是做选择（选A还是选B，取最大还是取最小）。Example：$dp[i] = dp[i-1] + dp[i-2]$。


3. 初始化 Base Case (Initialization)

核心问题：最开始的那几个值是多少？（防止数组越界）。

Tip：通常是 dp[0]、dp[1] 或者边界条件。 Example：dp[0]=1（原地不动算一种？），dp[1]=1。

4. 遍历顺序 (Traversal Order)

核心问题：是从前向后遍历，还是从后向前？是先遍历物品，还是先遍历背包？

Tip：通常取决于你的递推公式依赖于哪里的值。Example：因为 dp[i] 依赖 dp[i-1]，所以必须从前向后（$i$ 从 small 到 large）。

5. 空间优化 (Space Optimization) (可选) 

核心问题：如果只依赖前两个状态，真的需要开一个 $O(N)$ 的数组吗？

Tip：滚动数组（Rolling Array）技巧，将空间降为 $O(1)$。滚动数组指：用变量代替数组，或者用一维数组反复覆写来代替二维数组。


# Leetcode 上的 DP 题目

1. https://leetcode.com/problems/climbing-stairs/ (easy)
2. https://leetcode.com/problems/longest-increasing-subsequence/description/ (medium)
3. https://leetcode.com/problems/maximum-subarray/ (medium)  **


心得：多找找构建转移矩阵的感觉；** 表示第一遍没做出来的题