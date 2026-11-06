# 盘卷进入区域事件（HTTP `events`）方案

> 状态：已实现（HTTP 服务），单测 `enter_event_test` 全部通过
> 范围：HTTP 服务（`tedai_juanqu_service` + `tedai_juanqu_pipeline`），**不改 DLL**

## 1. 目标

在 HTTP 响应中新增 `events` 数组，通知 C#：**某个相机负责的输送线区域有新盘卷进入**。

每次请求、**每个相机最多输出 1 条** `enter` 事件；后续业务由 C# 侧自行判断。

## 2. 参与范围

| 相机 | 区域 | `roi_id` | `liuzhi_id` |
|---|---|---|---|
| 0 | 不参与 | — | — |
| 1 / 2 / 3 | 输送线内 | `inside` | 1 |
| 4 / 5 | 输送线外 | `outside` | 2 |

- 取各相机**最终结果列表**（已排序、已去重、已做分类过滤）
- cam4/5 的 `liuzhi_id=2` 起终点配置后期补齐；缺失 qishi 时走回退逻辑

## 3. 判定规则

### 3.1 数据选择

- 位置来源：cam3 的 `inside` 经过透视变换（x = 序号 + 50、y = 物理毫米），因此**入口判定使用变换前的 `inside_pixel`**；事件坐标仍输出与响应 `panjuan_location_information` 一致的那份
- 其它相机直接用输出框（像素坐标）

### 3.2 处理流程（每相机 + 每区域，每次请求）

```
输入：本帧区域框列表 cur[]，区域配置 qishi / zhongzhi

1) 连续性匹配
   与上一次请求保存的 prev[] 做一对一最近邻匹配（中心距离 < 150 px）
   匹配上的记为"老卷"，未匹配的记为"新卷"

2) 去重缓存过滤
   新卷若落在"最近 5 次请求已上报位置"的 120 px 内 → 丢弃（防漏检后重复上报）

3) 入口判定（仅新卷）
   若配置了 qishi / zhongzhi：
       d = |center - qishi|
       t = ((P - qishi)·(zhongzhi - qishi)) / |zhongzhi - qishi|²
       条件：d < 120  或  (0 <= t < 0.25)
   若未配置（如 cam4 现状）：
       条件：新出现即算进入

4) 每相机最多 1 条
   在满足条件的候选中取"到 qishi 距离最小"的一个
   （无 qishi 时取 y 最小的一个）
   其余候选只建立跟踪，不发事件

5) 更新状态
   prev[] = cur[]
   已上报位置写入缓存（带请求序号，超过 5 次请求淘汰）
   连续 2 次请求未匹配上的旧轨迹淘汰

6) 基线
   每个相机第一次请求、或距上次处理该相机超过 5 秒
   → 只建立基线，不发事件（避免启动 / 断流后刷一串事件）
```

### 3.3 事件顺序

每相机至多一条，`events` 内按 `camera_id` 升序排列。

## 4. 接口变更（HTTP 响应）

### 4.1 变更前

```json
{
  "station_id": "line-1",
  "read_message_flag": "OK",
  "all_results": [
    {
      "camera_id": 0,
      "read_picture_flag": 1,
      "result_picture_path": "result/juanqu/location/camera_00/20261106/20261106093015123_00.jpg",
      "single_camera_results": [
        {
          "roi_id": "inside",
          "roi_boxCount": 1,
          "panjuan_location_information": [
            {
              "panjuan_id": 0,
              "x": 350,
              "y": 450,
              "width": 130,
              "height": 110,
              "location": 0
            }
          ]
        },
        {
          "roi_id": "outside",
          "roi_boxCount": 0,
          "panjuan_location_information": []
        },
        {
          "roi_id": "collect",
          "roi_boxCount": 0,
          "panjuan_location_information": []
        }
      ]
    },
    {
      "camera_id": 5,
      "read_picture_flag": 1,
      "result_picture_path": "result/juanqu/location/camera_05/20261106/20261106093015124_05.jpg",
      "single_camera_results": [
        {
          "roi_id": "inside",
          "roi_boxCount": 0,
          "panjuan_location_information": []
        },
        {
          "roi_id": "outside",
          "roi_boxCount": 1,
          "panjuan_location_information": [
            {
              "panjuan_id": 0,
              "x": 400,
              "y": 500,
              "width": 120,
              "height": 100,
              "location": 0
            }
          ]
        },
        {
          "roi_id": "collect",
          "roi_boxCount": 0,
          "panjuan_location_information": []
        }
      ]
    }
  ]
}
```

说明：

- 当前输出只含 `camera_id` / `read_picture_flag` / `result_picture_path` / `single_camera_results`
- 每个相机固定三个 ROI：`inside`、`outside`、`collect`
- 盘卷字段只有 `panjuan_id` / `x` / `y` / `width` / `height` / `location`，没有 `coil_id`、`identity_status`
- `result_picture_path` 规则：`<result_dir>/camera_XX/YYYYMMDD/YYYYMMDDHHMMSSmmm_XX.jpg`
- 示例只列 camera0、camera5；实际请求里有多少相机就返回多少个（0–5）

### 4.2 变更后

```json
{
  "station_id": "line-1",
  "read_message_flag": "OK",
  "all_results": [ "... 与 4.1 完全相同，不新增/不修改任何字段 ..." ],
  "events": [
    {
      "event_id": "evt-8f3c2a61-4b0e-4f1d-9a58-2f6d0c9a1b77",
      "type": "enter",
      "camera_id": 1,
      "roi_id": "inside",
      "liuzhi_id": 1,
      "x": 1750,
      "y": 180,
      "width": 130,
      "height": 110
    }
  ]
}
```

- `events` 字段**始终存在**，无事件时为 `[]`
- 其余字段（`station_id` / `read_message_flag` / `all_results` 内部结构）**完全不变**

### 4.3 `events[]` 字段说明

| 字段 | 类型 | 生成方 | 含义 |
|---|---|---|---|
| `event_id` | string | HTTP | `evt-<UUID v4>`，每次事件唯一，C# 可用于去重 |
| `type` | string | HTTP | 固定 `"enter"` |
| `camera_id` | int | HTTP | 发生事件的相机 |
| `roi_id` | string | HTTP | `inside`（cam1-3）/ `outside`（cam4-5） |
| `liuzhi_id` | int | HTTP | 取自 ROI 配置：1 或 2 |
| `x` / `y` | int | HTTP | 进入瞬间框中心坐标，与该相机 `panjuan_location_information` 同源 |
| `width` / `height` | int | HTTP | 框宽高 |

### 4.4 结果图标注（调试）

`events.draw_debug: true` 时，保存的结果图上会额外绘制：

| 元素 | 含义 |
|---|---|
| 品红虚线 + 红/蓝端点 | `qishi → zhongzhi` 轴线与起终点 |
| 品红圆 | 以 `qishi` 为圆心、`entry_radius_px` 为半径的距离门槛 |
| 绿色粗框 + `ENTER d=.. t=.. rule=..` | 本帧上报的进入盘卷 |
| 灰色细框 + `no-enter` | 未通过入口判定的新框 |
| 黄色细框 + `dup` | 被去重缓存拦下的新框 |
| 左下角面板 | new / tracked / dup 数量、enter 结论、qishi/zhongzhi、radius / t_max、event_id |

说明：cam3 的标注使用检测像素坐标（`inside_pixel`），与响应中的物理坐标不同。

## 5. 配置项（`config.yaml`）

```yaml
events:
  enable: true            # 总开关
  draw_debug: true        # 在结果图上绘制进入状态与判定信息
  entry_radius_px: 120    # 到 qishi 的距离阈值
  entry_t_ratio: 0.25     # 沿 qishi->zhongzhi 轴的投影比例上限
  dedup_frames: 5         # 已上报位置缓存的请求数
  track_timeout_ms: 5000  # 相机超过该时长未参与请求则重建基线
  match_radius_px: 150    # 跨请求连续性匹配阈值
  max_missed: 2           # 连续未匹配多少次后淘汰轨迹
```

未配置时使用上述默认值。

## 6. 代码改动清单

| 文件 | 改动 |
|---|---|
| `src/nisco_project/tedai_juanqu/reel_types.h` | 新增 `ReelEnterEvent` 结构（event_id / type / camera_id / roi_id / liuzhi_id / x / y / w / h） |
| `src/nisco_project/tedai_juanqu/reel_region_tracker.h/.cpp`（新增） | 跨请求状态：连续性匹配、入口判定、每相机取一条、去重缓存、UUID |
| `src/nisco_project/tedai_juanqu/tedai_juanqu_pipeline.h/.cpp` | 暴露每个相机的 ROI 元信息（`liuzhi_id`、`qishi`、`zhongzhi`）；`process()` 后调用 tracker；提供 `lastEvents()` |
| `src/nisco_project/tedai_juanqu/tedai_juanqu_service.cpp` | 响应 `all_results` 后追加 `events` |
| `src/nisco_project/tedai_juanqu/file_utils.h/.cpp` | 解析 `events` 配置段 |
| `src/CMakeLists.txt` | 新源文件加入 `NISCO_PROJECT_SOURCES` |
| `test/tedai_juanqu/config.yaml` | 增加 `events` 配置段（默认值） |
| `test/tedai_juanqu/enter_event_test.cpp`（新增） + `test/CMakeLists.txt` | 合成数据单测 |
| `doc/http_enter_events.md`（新增） | 接口说明（给 C# 侧） |

## 7. 测试与验证

### 7.1 单元测试（不依赖模型）

| 场景 | 期望 |
|---|---|
| 第一次请求，区域内已有卷 | `events=[]`（建基线） |
| 第二次请求，同一卷 | `events=[]` |
| 第二次请求，入口附近新增 1 个卷 | 1 条 enter，坐标正确 |
| 入口附近同时出现 2 个新卷 | 只出 1 条（离 qishi 最近的那个） |
| 区域中间突然出现一个框 | `events=[]` |
| 漏检 1 次后同一卷再次出现 | `events=[]`（去重缓存生效） |
| cam4 未配置 qishi | 回退为"新出现即进入"，取 y 最小 |
| cam3 | 使用 `inside_pixel` 判定，事件坐标与响应一致 |

### 7.2 端到端

1. 启动 HTTP 服务，POST 两次同一组图 → `events` 均为 `[]`
2. 构造 / 使用包含新卷的图像再次 POST → 恰好 1 条，字段完整
3. 确认 `all_results` 与改动前逐字段一致（除新增 `events`）

### 7.3 编译

`cmake --build build --config Release --target tedai_juanqu`（HTTP 服务可执行）+ 单测目标。

## 8. 风险与假设

1. "其余新卷静默"是既定行为：每相机只报最近入口的一个，其余仅建跟踪
2. 状态仅在内存：服务重启后第一次请求只建基线，不补发事件
3. 请求间隔过大（> 5s）会重建基线，可能漏一次进入
4. cam4 的 qishi 未配置，暂时按"新出现即进入"处理，配置补齐后自动切换
5. cam3 使用像素坐标判定、物理坐标输出，事件坐标与该相机列表一致
6. `events` 之外的字段一律不改，保证 C# 旧解析兼容
