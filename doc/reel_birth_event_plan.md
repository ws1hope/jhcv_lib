# 盘卷出生事件接口计划（JHMVDetect.dll / tedai_juanqu camera0）

> 状态：已实现（对接说明见 `doc/reel_birth_event.md`）

## 1. 背景与问题

C# 侧反馈：**上报坐标时无法确定这条坐标对应哪个盘卷号；画面里同时存在两个盘卷时必然对不上。**

当前 DLL 能提供的"号"只有三种形式：

| 现有能力 | 实际拿到的内容 | 多卷时的问题 |
|---|---|---|
| `boxs_exist_region_current[i].order_number` | 当天序号（如 7） | 不是完整 ID，且结构体 48 字节已冻结，塞不进 `long long` |
| `MV_SDK_GetLastReelId` / `MV_SDK_GetLastReelIdValue` | 只有一个"最近出生"的值 | 同帧出生两个卷时前一个被覆盖丢失，C# 只拿得到最后一个 |
| `MV_SDK_PopReelLeaveEvent` | 完整 ID + 坐标 + 工位 + 时间 | 离开侧没问题，但出生/在途阶段没有对应事件 |

结论：**缺的是"出生那一刻，把号和坐标一起交出去"的通道**，不是赋号逻辑本身有问题。

## 2. 目标与范围

### 本次做

1. 新增 `ReelBirthEvent` 结构体与 `MV_SDK_PopReelBirthEvent` 导出；
2. 事件内同时带 `reel_id`、`order_number`、`station_number`、出生帧坐标、出生时间；
3. C# 用事件自建 `order_number → reel_id` 对照表，覆盖出生帧之后的每一帧。

### 本次不做（明确排除）

- 不新增 `MV_SDK_GetReelIdByOrder` / `MV_SDK_GetReelIdValueByOrder`（由事件里的 `order_number` 替代）；
- 不修改 `reelBoxMatch` / `InputParamsReelExportTargetTrack` / `OutputResultReelExportTargetTrack` 任何字段与大小（C# 结构体布局零改动）；
- 不修改 `MV_SDK_GetLastReelId` / `MV_SDK_GetLastReelIdValue` 的现有语义（保留兼容，文档标注为旧接口）；
- 不做跟踪稳定性加固（漏检重号、回退丢号、快速移动串号），见第 8 节。

## 3. 接口设计

### 3.1 结构体

```c
typedef struct ReelBirthEvent {
    long long reel_id;        // 完整号，如 20261106007
    int order_number;         // 当天序号，与输出框 order_number 同源
    int station_number;       // 1 / 2
    int x;                    // 出生帧中心 x
    int y;                    // 出生帧中心 y
    int width;                // 出生帧框宽
    int height;               // 出生帧框高
    long long birth_time_ms;  // 出生时刻，Unix 毫秒（取 Detect_2 该帧时间）
} ReelBirthEvent;             // 默认 Pack 下 40 字节
```

- 与 `ReelLeaveEvent` 同为 40 字节、同样字段风格，便于 C# 复用已有的 marshalling 习惯；
- `order_number` 是本方案的关键字段：有了它，C# 不需要额外的"序号反查"接口。

### 3.2 函数

```c
// 取一条出生事件：1 = 取到，0 = 队列已空
int REEL_TRACK_API REEL_TRACK_STDCALL MV_SDK_PopReelBirthEvent(ReelBirthEvent* evt);
```

新增导出共 **1 个函数 + 1 个结构体**。

## 4. 事件语义

| 项目 | 说明 |
|---|---|
| 触发时机 | 盘卷在存在区内被判定为"新生"并赋号的那一帧（`assignNewReelOrder` 内，号已确定后入队） |
| 覆盖范围 | 仅工位 1 / 2 的正常盘卷；异常卷、回退卷（`order_number = 0`）不产生事件 |
| 同帧多卷 | 每卷一条，互不覆盖；入队顺序即赋号顺序（y 降序：图像下方先拿号） |
| 去重 | 同一个 `reel_id` 只会出生一次，因此只产生一条事件 |
| 队列 | FIFO，容量 64；满了丢最旧并写日志 `[ReelBirth] drop oldest (queue full, cap=64)` |
| 生命周期 | `Init` / `Destroy` 时清空队列（与离开事件一致） |
| 取用线程 | 与 `Detect_2` 同一线程、同一把 `g_mutex` 保护，不需要额外同步 |
| 坐标含义 | 出生那一帧的中心点与框尺寸，与 `boxs_exist_region_current` 的 `x/y/width/height` 同源同单位 |

## 5. 内部实现改动

### 5.1 pipeline 层（`src/nisco_project/tedai_juanqu/reel_track_pipeline.h/.cpp`）

1. 新增 `ReelBirthEventInfo`（镜像 C ABI 结构体，不含时间戳）：

```cpp
struct ReelBirthEventInfo {
    long long reel_id = 0;
    int order_number = 0;
    int station_number = 0;
    int x = 0, y = 0, width = 0, height = 0;
};
```

2. `assignNewReelOrder(reelBoxMatch& box)` 内，赋值完成后追加：

```cpp
ReelBirthEventInfo info;
info.reel_id = last_reel_id_value_;
info.order_number = box.order_number;
info.station_number = box.station_number;
info.x = box.x; info.y = box.y;
info.width = box.width; info.height = box.height;
birth_events_.push_back(info);
```

调用点共 4 处（上一帧空/当前 1 个、上一帧空/当前 ≥2 个、上一帧非空且匹配失败），这些位置在调用前都已填好 `x/y/width/height/station_number`，无需额外补字段。

3. 新增 `std::vector<ReelBirthEventInfo> takeBirthEvents();`（取走即清空，与 `takeLeaveEvents()` 一致），成员 `std::vector<ReelBirthEventInfo> birth_events_;`，并在 `init()` 中清空。

4. 逻辑无变化：出生事件的记录点与现有赋号点完全重合，不新增判定分支。

### 5.2 DLL 层（`src/cv_infer/reel_track_dll.h/.cpp/.def`）

1. `.h`：加入 `ReelBirthEvent` 结构体、`MV_SDK_PopReelBirthEvent` 声明，并在文件头的导出清单注释里补充该接口。
2. `.cpp`：
   - 新增 `const size_t kMaxBirthEvents = 64;` 与 `std::deque<ReelBirthEvent> g_birth_events;`（与 `g_leave_events` 并列）；
   - `Detect_2` 中在 `takeLeaveEvents()` 附近追加：取 `takeBirthEvents()`，补 `birth_time_ms`（同一次 `system_clock` 取时间），超限丢最旧并写日志 `[ReelBirth] drop oldest ...`，入队后写日志 `[ReelBirth] id=... order=... station=... center=(...) pending=...`；
   - `Init` / `Destroy` 中 `g_birth_events.clear();`；
   - 实现 `MV_SDK_PopReelBirthEvent`：加锁、判空、拷出、出队、写日志 `[ReelBirth] pop id=... pending=...`。
3. `.def`：追加 `MV_SDK_PopReelBirthEvent` 一行导出。

### 5.3 测试程序（`test/tedai_juanqu/dll_test.cpp`）

每次 `Detect_2` 返回后：

```cpp
ReelBirthEvent be;
while (MV_SDK_PopReelBirthEvent(&be) == 1) {
    printf("  birth id=%lld order=%d station=%d center=(%d,%d) size=%dx%d\n",
           be.reel_id, be.order_number, be.station_number,
           be.x, be.y, be.width, be.height);
}
```

同时在每帧打印 `order_number` 后追加 `MV_SDK_GetLastReelId` 对照，便于现场核对两者一致性。

### 5.4 文档（`doc/reel_birth_event.md`，实现时新建）

内容与本文第 6 节一致：接口说明 + C# 接入步骤 + 日志排查 + 已知限制。风格对齐 `doc/reel_leave_event.md`。

## 6. C# 接入（交给对方照做）

### 6.1 声明

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

### 6.2 每帧调用顺序

```csharp
IntPtr outImg = MV_SDK_ReelExportTargetTrack_Detect_2(buf, w, h, ch, p, out var outRes);

// 1) 先把出生事件取干净（必须 while，一帧可能 0/1/多条）
ReelBirthEvent be;
while (MV_SDK_PopReelBirthEvent(out be) == 1)
{
    idByOrder[be.order_number] = be.reel_id;   // 自建对照表
    OnReelBorn(be.reel_id, be.station_number, be.x, be.y,
               be.width, be.height, be.birth_time_ms);
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

离开区同样用 `boxs_leave_region_current[i].order_number` 查表。

### 6.3 必须遵守的三条

1. **每帧都取**：紧接着 `Detect_2` 返回之后取，不要累积到几十帧后再取（队列 64 条会丢最旧的）；`Detect_2` 返回 NULL 时也取一次清理积压。
2. **上一帧框整块回传**：`input_params.boxs_exist_region_previous` 必须是整结构体拷贝（`prev[i] = cur[i]`），不能只抄 `x/y/w/h`；否则 `order_number` 被清 0，号码继承断掉会导致重复赋号。
3. **对照表要清理**：跨天（序号从 1 重新开始）和重新 `Init` 时清空 `idByOrder`，否则会把今天的序号对成昨天的号。

### 6.4 旧接口处理

- `MV_SDK_GetLastReelId` / `GetLastReelIdValue` 保留可用，但**不要再用于坐标绑定**（只返回最近一个号）；
- 现有离开事件 `MV_SDK_PopReelLeaveEvent` 不受影响，继续使用。

## 7. 已知限制（需向 C# 说明）

1. **重启前已在区域内的卷没有出生事件**，查不到号（业务只关心新出生卷时无影响；若必须补，才需要再加 `GetReelIdValueByOrder`）。
2. **同帧多卷的事件顺序是 y 降序**，不等于工艺先后；请用 `station_number` 与坐标判断，不要依赖顺序。
3. **漏检/匹配失败会重号**（同一物理卷拿到两个 `reel_id`），属于跟踪逻辑问题，不在本次范围。
4. 事件仅存在于内存，进程重启清空；队列溢出会丢最旧事件。
5. 跨零点短暂窗口内，旧序号理论上可能与当天同序号冲突（单帧级，概率极低）。

## 8. 后续可选增强（不在本次实现）

| 增强 | 解决的问题 | 触发条件 |
|---|---|---|
| `MV_SDK_GetReelIdValueByOrder(order)` | 重启后在区域老卷查号、事件丢失补救 | C# 提出需要时再加 |
| 跟踪容忍漏检（多帧轨迹） | 漏检一帧导致重号 | 现场出现重号 |
| 回退不清零 / 离开区保留号 | 退回来丢号 | 现场出现回退丢号 |
| 匹配速度预测 + 双向校验 | 两卷靠太近串号 | 现场出现串号 |

## 9. 验证计划

1. 编译：
   - `cmake --build build --config Release --target jhcv_lib`
   - `cmake --build build --config Release --target reel_track_dll`
2. 结构体尺寸：C# 侧 `Marshal.SizeOf == 40`；C 侧可加 `static_assert(sizeof(ReelBirthEvent) == 40)`。
3. 运行 `test/tedai_juanqu/dll_test`（需模型与图像，见 `.vscode/launch.json` 的 PATH）：
   - 单卷出生 → 一条事件，`order=1`、`id=当天日期+001`；
   - 同帧双卷（构造或现场录像）→ 两条事件，`order` 不同、`id` 不同、坐标各自对得上；
   - 离开区事件仍正常，`reel_id` 与出生事件一致。
4. 日志核对：`[ReelBirth] id=... order=... center=(...)` 与 `[Draw] ... labels=...` 中的号码一致。

## 10. 变更文件清单

| 文件 | 改动 |
|---|---|
| `src/cv_infer/reel_track_dll.h` | `ReelBirthEvent` + `MV_SDK_PopReelBirthEvent` 声明、导出清单注释 |
| `src/cv_infer/reel_track_dll.cpp` | 出生事件队列（FIFO/64/丢最旧/日志）、`Detect_2` 入队、`Init`·`Destroy` 清空、pop 实现 |
| `src/cv_infer/reel_track_dll.def` | 追加 1 行导出 |
| `src/nisco_project/tedai_juanqu/reel_track_pipeline.h` | `ReelBirthEventInfo`、`takeBirthEvents()`、成员 `birth_events_` |
| `src/nisco_project/tedai_juanqu/reel_track_pipeline.cpp` | `assignNewReelOrder()` 入队、`init()` 清空、`takeBirthEvents()` 实现 |
| `test/tedai_juanqu/dll_test.cpp` | 每帧 drain 出生事件并打印 |
| `doc/reel_birth_event.md`（实现时新建） | C# 对接文档 |

无 ABI 破坏：不新增/修改任何既有结构体字段，仅新增导出函数；新增文档无编码问题（`doc` 下为 UTF-8），`reel_track_dll.cpp` 已有 `/utf-8`（中文注释可行）。
