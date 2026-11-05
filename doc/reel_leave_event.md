# 盘卷离开事件接口说明（JHMVDetect.dll）

> 状态：DLL 侧已实现，C# 侧待接入。

## 1. 目的

DLL 在盘卷越过"离开分钢线"时产生一条事件，C# 每帧调用 `Detect_2` 后取走事件即可，无需 HTTP、无需额外线程。

## 2. 盘卷号（reel_id）

- 格式：`YYYYMMDD` + 3 位当天序号，**纯数字、无下划线**
- 示例：`20261104007`
- 类型：`long`（64 位整数）
- 当天序号跨进程重启续编，跨天从 `001` 重新开始

## 3. 新增接口

```c
typedef struct ReelLeaveEvent {
    long long reel_id;        // 20261104007
    int station_number;       // 1 / 2
    int x, y;                 // 离开瞬间的中心坐标
    int width, height;
    long long leave_time_ms;  // Unix 毫秒时间戳
} ReelLeaveEvent;

// 取一条事件：1 = 取到，0 = 队列为空
int  MV_SDK_PopReelLeaveEvent(ReelLeaveEvent* evt);

// 最近一次出生的盘卷号（纯数字），无出生返回 0
long long MV_SDK_GetLastReelIdValue(void);
```

原有接口（`Init` / `Detect_2` / `Free` / `GetLastError` / `MV_SDK_GetLastReelId`）签名不变。

## 4. 事件语义

| 项目 | 说明 |
|---|---|
| 触发时机 | 盘卷中心首次越过 `set_leave_fengang_line`，进入离开区域 |
| 去重 | 同一个 `reel_id` 只产生一次事件；回退后再次离开不重复产生 |
| 不产生事件 | 未赋过号的盘卷、异常卷、回退途中 |
| 队列容量 | 64 条；C# 长期不取时丢弃最旧的，并写调试日志 |
| 生命周期 | `Init` / `Destroy` 时清空队列和去重记录 |

## 5. C# 接入步骤

1. 增加 `ReelLeaveEvent` 结构体和上述两个函数声明
   - 结构体字段顺序、类型、总大小必须与 DLL 完全一致
2. 每帧 `Detect_2` 返回后循环取事件（详见第 5.1 节）
3. 修改号码解析：由 `20261104_007` 改为纯数字 `20261104007`（或改用 `MV_SDK_GetLastReelIdValue`）

### 5.1 每帧取事件的具体做法

调用时机：**紧接着本次 `Detect_2` 返回之后**，在调用下一帧之前。同一帧产生的事件在 `Detect_2` 返回时已经进入队列，此时取到的就是这一帧的结果。

```
while (MV_SDK_PopReelLeaveEvent(out evt) == 1)
{
    // 1. 取到一条：evt.reel_id / station_number / x / y / leave_time_ms 已填好
    // 2. 建议只做拷贝或入队，重活（写库、发消息、UI 刷新）交给业务线程
}

// 取不到时返回 0，循环结束；再继续处理本帧结果图
```

规则：

- 用 `while`，不要用 `if`：一帧可能有 0 条、1 条或多条（多个盘卷同帧离开）
- 循环条件必须是 `== 1`；返回 0 只代表"队列空了"，不是错误
- 顺序为先进先出，即盘卷越过分钢线的先后顺序
- 即使 `Detect_2` 返回 NULL（本帧检测失败）也可以取一次，把历史积压清掉
- 不要在别的线程里和 `Detect_2` 同时取；固定在调用 `Detect_2` 的线程里取最省心
- 取到后尽快返回，避免单帧业务耗时过长影响下一帧；事件本身不会丢失
- 若某天没人取：队列最多 64 条，满了丢最旧的，并写日志 `[ReelEvent] drop oldest`

## 6. 注意事项

- DLL 为 x64，C# 进程必须为 64 位
- 若 C# 暂不接入：检测与画图不受影响，仅事件无人消费（队列满后丢弃）
- 事件在 `Detect_2` 的调用线程内产生，取事件与检测串行，不影响帧耗时
- 去重记录只在内存中，不跨进程：若重启时盘卷仍停在离开区，可能会再上报一次
- 现场排查可用日志：
  - `[Init] build="..."` 确认加载的是哪一版 DLL
  - `[ReelEvent] leave id=... station=... center=(...)` 确认事件是否产生
  - `[Draw] exist=.. leave=.. with_id=.. labels=..` 确认号码是否传到绘制（`with_id=0` 表示没有号码可画）

## 7. 现场验证

1. 让盘卷越过离开分钢线，观察日志：
   `[ReelEvent] leave id=20261104007 station=1 center=(1425,1166)`
2. C# 应能取到对应事件，且同一盘卷只出现一次
3. 若无事件：检查是否已产生盘卷号、是否越过 `set_leave_fengang_line`
