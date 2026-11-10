# HTTP 盘卷进入区域事件（`events`）接口说明

> 适用服务：`tedai_juanqu`（`POST /tedai_panjuan_detect`）
> 本次只新增响应字段 `events`，其余字段一律不变

## 1. 用途

通知调用方：**某个相机负责的输送线区域有新盘卷进入**。

每次请求、**每个相机的每个区域最多 1 条** `enter` 事件；后续业务判断由调用方完成。

## 2. 参与范围

| 相机 | 区域 | `roi_id` | `liuzhi_id` |
|---|---|---|---|
| 0 | 不参与 | — | — |
| 1 / 2 / 3 | 输送线内 | `inside` | 1 |
| 4 | 输送线内 + 输送线外 | `inside` / `outside` | 1 / 2 |
| 5 | 输送线外 | `outside` | 2 |

> camera4 的 `inside`（liuzhi 1）检测要求 `camera_liuzhi_roi.json` 中 camera4
> 配置了 liuzhi_id=1 的 ROI；未配置时该区域不产生事件。

## 3. 响应变更

### 3.1 变更前

```json
{
  "station_id": "line-1",
  "read_message_flag": "OK",
  "all_results": [ { "camera_id": 0, "...": "..." } ]
}
```

### 3.2 变更后

```json
{
  "station_id": "line-1",
  "read_message_flag": "OK",
  "all_results": [ { "camera_id": 1, "...": "..." } ],
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

- `events` **始终存在**，无事件时为 `[]`
- `all_results` 内部结构完全不变

### 3.3 字段说明

| 字段 | 类型 | 含义 |
|---|---|---|
| `event_id` | string | `evt-<UUID v4>`，每条唯一，可用于去重 |
| `type` | string | 固定 `"enter"` |
| `camera_id` | int | 发生事件的相机 |
| `roi_id` | string | `inside`（cam1-4）/ `outside`（cam4-5） |
| `liuzhi_id` | int | 取自 ROI 配置：1 或 2 |
| `x` / `y` | int | 进入瞬间框中心坐标（像素），与该相机 `panjuan_location_information` 同源 |
| `width` / `height` | int | 框宽高（像素） |

## 4. 事件语义

| 项目 | 说明 |
|---|---|
| 触发 | 新盘卷出现在区域入口附近（到 `qishi` 距离 < 120px，或沿 `qishi→zhongzhi` 轴投影 `0 <= t < 0.25`） |
| 每相机每区域上限 | 1 条；多个候选时取离 `qishi` 最近的一个，其余只建跟踪 |
| 基线 | 服务重启后第一次请求、或该相机超过 5 秒未参与请求 → 只建基线，不发事件 |
| 去重 | 同一盘卷持续在区域内只报一次；漏检后重现若在已上报位置 120px 内且未超过 5 次请求，不重复上报 |
| 未配置起终点 | 该区域退化为"新出现即进入"，多个候选取 `y` 最小（当前 camera4 属于这种情况） |

## 5. 注意事项

- 仅新增字段，旧解析逻辑不受影响
- `events` 可能为空数组，调用方需按空数组处理
- 各相机的 `panjuan_location_information` 与事件坐标**同为检测像素坐标**（camera3 原透视变换/坐标偏移映射已于 2026-11 取消）
- 事件状态只在服务内存中，不跨重启
- 可配置项见 `config.yaml` 的 `events` 段（阈值、去重次数、超时等）

## 6. 结果图标注（调试）

`events.draw_debug: true` 时，服务保存的结果图上会画出：

- `qishi → zhongzhi` 轴线（品红虚线）与起终点标记
- 以 `qishi` 为圆心、`entry_radius_px` 为半径的距离门槛圆
- 本帧上报的进入盘卷：绿色粗框 + `ENTER d=.. t=.. rule=..`
- 未通过入口判定的新框：灰色细框 + `no-enter`
- 被去重缓存拦下的新框：黄色细框 + `dup`
- 左下角信息面板：new/tracked/dup 数量、enter 结论、qishi/zhongzhi、radius/t_max、event_id

该标注只影响结果图，不影响 HTTP 响应内容；不需要时把 `draw_debug` 设为 `false`。
