# 盘卷出生事件接口说明（JHMVDetect.dll）

> 状态：DLL 侧已实现，C# 侧待接入。
> 适用：`tedai_juanqu` camera0 工位（盘卷出口跟踪）。

## 1. 目的

DLL 在盘卷"出生"（第一次被判定为工位 1/2 的正常盘卷并赋号）时产生一条事件，事件里**同时带盘卷号和当时的坐标**，C# 每帧调用 `Detect_2` 后取走事件即可。

这样即使画面里同时存在两个盘卷，坐标和号也是一对一的，不需要再用 `MV_SDK_GetLastReelId` 去猜。

## 2. 盘卷号（reel_id）

- 格式：`YYYYMMDD` + 3 位当天序号，**纯数字、无下划线**
- 示例：`20261106007`
- 类型：`long`（64 位整数）
- 当天序号跨进程重启续编，跨天从 `001` 重新开始
- `order_number` 是同一天内的序号（7），`reel_id` 是完整号（20261106007）

## 3. 新增接口

```c
typedef struct ReelBirthEvent {
    long long reel_id;        // 20261106007
    int order_number;         // 当天序号，与输出框 order_number 一致
    int station_number;       // 1 / 2
    int x, y;                 // 出生那一帧的中心坐标
    int width, height;        // 出生那一帧的框
    long long birth_time_ms;  // Unix 毫秒时间戳（本帧时间）
} ReelBirthEvent;             // 40 字节

// 取一条出生事件：1 = 取到，0 = 队列为空
int MV_SDK_PopReelBirthEvent(ReelBirthEvent* evt);
```

原有接口（`Init` / `Detect_2` / `Free` / `GetLastError` / `GetLastReelId` /
`GetLastReelIdValue` / `PopReelLeaveEvent`）签名与语义均不变。

## 4. 事件语义

| 项目 | 说明 |
|---|---|
| 触发时机 | 盘卷在存在区内被判定为新生并赋号的那一帧 |
| 覆盖范围 | 仅工位 1 / 2 的正常盘卷；异常卷、回退卷（`order_number = 0`）不产生事件 |
| 同帧多卷 | 每卷一条，互不覆盖；两条事件的 `reel_id` 不同、坐标各自对应自己的框 |
| 事件顺序 | 入队顺序即赋号顺序，为 **y 降序**（图像下方先拿号），**不等于工艺先后** |
| 去重 | 同一个 `reel_id` 只会出生一次，因此只产生一条事件 |
| 坐标含义 | 出生帧的中心点与框尺寸，与 `boxs_exist_region_current` 的 `x/y/width/height` 同源 |
| 队列容量 | 64 条；C# 长期不取时丢弃最旧的，并写日志 `[ReelBirth] drop oldest` |
| 生命周期 | `Init` / `Destroy` 时清空队列 |

## 5. C# 接入步骤

### 5.1 声明

```csharp
[StructLayout(LayoutKind.Sequential)]      // 使用默认 Pack，不要改成 Pack=1
public struct ReelBirthEvent
{
    public long reel_id;
    public int order_number;
    public int station_number;
    public int x;
    public int y;
    public int width;
    public int height;
    public long birth_time_ms;
}

[DllImport("JHMVDetect.dll", CallingConvention = CallingConvention.StdCall)]
public static extern int MV_SDK_PopReelBirthEvent(out ReelBirthEvent evt);
```

启动自检：`Debug.Assert(Marshal.SizeOf<ReelBirthEvent>() == 40);`

### 5.2 每帧调用顺序

调用时机：**紧接着本次 `Detect_2` 返回之后**，在调用下一帧之前。同一帧产生的事件在 `Detect_2` 返回时已经进入队列。

```csharp
// 序号 -> 完整号，跨帧保留；跨天和重新 Init 时清空
private readonly Dictionary<int, long> idByOrder = new Dictionary<int, long>();

IntPtr outImg = MV_SDK_ReelExportTargetTrack_Detect_2(buf, w, h, ch, p, out var outRes);

// 1) 先把出生事件取干净（必须 while：一帧可能 0/1/多条）
ReelBirthEvent be;
while (MV_SDK_PopReelBirthEvent(out be) == 1)
{
    idByOrder[be.order_number] = be.reel_id;
    OnReelBorn(be.reel_id, be.order_number, be.station_number,
               be.x, be.y, be.width, be.height, be.birth_time_ms);
}

// 2) 任意一帧的框都能配上自己的号
for (int i = 0; i < outRes.exist_region_current_frame_boxCount; i++)
{
    var b = outRes.boxs_exist_region_current[i];
    if (b.order_number <= 0) continue;                     // 回退卷无号
    if (idByOrder.TryGetValue(b.order_number, out long id))
        ReportPosition(id, b.station_number, b.x, b.y, b.width, b.height);
}
```

离开区同理：用 `boxs_leave_region_current[i].order_number` 查同一张表；或优先使用离开事件自带的 `reel_id`（`MV_SDK_PopReelLeaveEvent`）。

### 5.3 必须遵守的三条

1. **每帧都取**：紧接着 `Detect_2` 返回之后取，不要累积几十帧后再取（队列 64 条会丢最旧的）；`Detect_2` 返回 NULL 时也取一次，把积压清掉。
2. **上一帧框整块回传**：`input_params.boxs_exist_region_previous` 必须是整结构体拷贝（`prev[i] = cur[i]`），不能只抄 `x/y/w/h`。否则 `order_number` 被清 0，号码继承断掉，会出现同一盘卷重复赋号。
3. **对照表要清理**：跨天（序号从 1 重新开始）和重新 `Init` 时清空 `idByOrder`，否则今天的序号会被对成昨天的号。

### 5.4 旧接口处理

- `MV_SDK_GetLastReelId` / `MV_SDK_GetLastReelIdValue` 保留可用，但**不要再用它做坐标绑定**：它只返回"最近出生"的一个号，多卷时必然错。
- 离开事件 `MV_SDK_PopReelLeaveEvent` 不受影响，继续使用。

## 6. 注意事项

- DLL 为 x64，C# 进程必须为 64 位
- 事件只在内存中，进程重启清空；重启前已在区域内的盘卷没有出生事件（它们的号没有被重新创建）
- 队列溢出会丢最旧事件，正常每帧取不会发生
- 跨零点时序号重置，理论上存在单帧级的序号与前一天撞号的窗口（概率极低）
- 漏检或帧间位移过大时同一盘卷可能被重新赋号（产生两条出生事件），属于跟踪逻辑问题，不在本接口范围

## 7. 现场排查

日志中相关关键字：

| 日志 | 含义 |
|---|---|
| `[ReelBirth] id=... order=... station=... center=(...) pending=...` | 出生事件已入队 |
| `[ReelBirth] pop id=... order=... pending=...` | C# 已取走一条 |
| `[ReelBirth] drop oldest (queue full, cap=64)` | C# 长时间没取，丢了最旧的一条 |
| `[Draw] exist=.. leave=.. with_id=.. labels=..` | 号码是否传到绘制（`with_id=0` 表示没号可画） |
| `[ReelId] state loaded/saved: date=... seq=...` | 当天序号持久化状态 |

## 8. 冒烟验证（已有测试程序）

`test/tedai_juanqu/dll_test.cpp` 会在每帧 `Detect_2` 后打印出生事件：

```
birth_event: reel_id=20261106001 order=1 station=2 center=(2474,566) size=170x157 time_ms=...
birth_event: reel_id=20261106002 order=2 station=1 center=(1844,214) size=172x73 time_ms=...
```

同一帧两条事件、号不同、坐标各自对应自己的框，即说明"多卷绑定"已生效；第二帧不再重复产生事件，说明号码继承正常。
