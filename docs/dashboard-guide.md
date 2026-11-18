# Dashboard guide

What every part of the dashboard shows, where the number comes from, and how to read it.

Open it with `scripts/dashboard.sh`, then go to http://localhost:8502. It only reads the database; nothing you do in it changes any data. All times are IST.

## How the system works, in four steps

1. **Capture.** Every 15 minutes one photo is taken from each camera (admin cabin, parking). Each 15-minute step is called a **slot**.
2. **Read.** Five models turn each photo into numbers: a detector that boxes objects, an outliner that traces each boxed object, a vision-language model (VLM) that describes the scene, an image-vector model that summarises the whole picture, and a description scorer that rates the photo against plain-English sentences.
3. **Table.** The numbers become one row per slot and one column per measured thing. A column is called a **dimension** or a **series**.
4. **Forecast.** Once an hour Chronos-2 reads the whole table and predicts the next 4 hours (16 slots) for a chosen set of series.

## Header

| Part | Meaning |
|---|---|
| Date and time (right) | The time the page was last refreshed. |
| Coloured dot and state word | **live**: the latest read photo is at most 20 minutes old. **late**: 20 to 60 minutes old. **stopped**: older than an hour, so capture or reading has stalled. **no photos**: nothing stored for that space. |
| photo 09:00 (10 min ago) | When the latest photo that has been read was taken, and its age. |
| 5 vehicles, 0 people | What the detector counted in that photo. Vehicles are shown for parking only. |
| forecast 09:00 | The time the newest stored forecast starts from. Forecasts are made on the hour. |

Below the header:

- **Space switch** (Admin cabin / Parking area): every tab shows the chosen space.
- **"showing ... up to ..."**: the newest slot in the table for that space.
- **"refreshes every minute"**: the page re-checks the database once a minute. It rebuilds its numbers only when a new photo or forecast has arrived.

## Tab 1: Now and next 4 hours

### Latest photo
The newest captured photo of the space, so the numbers can be compared with what the camera saw.

### The four tiles
The four **headline series** for the space: the ones named in the space's settings file, then the object types counted most often. Each tile keeps the same colour as its chart below.

| Part | Meaning |
|---|---|
| Name | What is measured and by which model, for example "people (detector)" or "motorcycle count (VLM)". |
| Large number | The value in the latest read photo. |
| +0.8 vs usual 0.2 | **usual** is the average of this series at the same clock time on earlier days (today is left out). The first number is how far the current value is above (+) or below (-) that. |

### Forecast table
One row for every series that is forecast for this space.

| Column | Meaning |
|---|---|
| series | The measured thing. |
| now | Its value in the latest read photo. |
| usual at this time | Average at this clock time on earlier days. |
| +1 h, +2 h, +4 h (10% to 90%) | The forecast for 1, 2 and 4 hours after the forecast start. The first number is the middle estimate: the true value is as likely to be above as below it. The bracket is the range the model gives an 80% chance of containing the true value (10% chance below, 10% chance above). A wide bracket means the model is unsure. |

Counts are forecast as fractions (1.5 people) because they are averages over what could happen, not a single guess.

### Series picker and History
- **Series**: which charts to draw. Any forecast series can be added.
- **History**: how much of the past each chart shows: 24 hours, 48 hours or 7 days.

### Charts
One chart per chosen series.

| Mark | Meaning |
|---|---|
| Solid line | Measured values, one point per slot. A break in the line is a slot with no usable photo. |
| Dashed line, same colour | The forecast's middle estimate for the next 4 hours. |
| Shaded band | The 10% to 90% range of the forecast. |
| Grey dotted line | The usual value at each clock time (earlier days only). Where the solid line is far from it, today is unusual. |
| Thin vertical line | The newest slot: measured values are to its left, the forecast to its right. |

Hover over a chart to read exact values. Headline series are blue, orange, green and violet; any other series you add is drawn in dark grey.

## Tab 2: Forecast vs actual

Shows how past forecasts compared with what was then measured.

- **Series**: the series to check.
- **Forecast made**: how far ahead the forecast was made: 1, 2 or 4 hours before the moment it predicts.

### Tiles

| Tile | Meaning |
|---|---|
| Forecasts checked | How many stored forecasts at this lead time now have a measured value to compare with. |
| Average miss | The average distance between the middle estimate and the measured value, in the series' own units (0.3 = off by 0.3 people on average). |
| Average measured value | The average of the measured values, to judge the size of the miss against. |
| Measured inside the shaded range | The share of measured values that fell inside the forecast's 10% to 90% range. About 80% is right. Well below means the ranges are too narrow (the model is overconfident); well above means they are wider than needed. |

### Chart
Dark line: what was measured. Blue line and band: what the forecast made that far ahead said for the same moment. The closer the two lines, the better. Blue points to the right of where the dark line ends are forecasts whose time has not come yet.

### All series together
Scores over every forecast series of the space, grouped by how far ahead the forecast looked.

| Column | Meaning |
|---|---|
| inputs given | The set of inputs the model received. `all` = the calendar plus every usable dimension. `calendar` = the calendar only (used before 18 Nov 2026, 08:15). |
| how far ahead | Lead time group: 0-1 h, 1-2 h, 2-4 h. |
| forecast runs | Number of hourly forecasts scored. |
| series | Number of series in those forecasts. |
| forecasts checked | Number of single predictions scored (runs x series x slots). |
| skill | How much smaller the miss is than the simplest guess, "nothing changes from the last value". 0 = no better than that guess, 1 = perfect, negative = worse. To pool series of different sizes, each series' miss is first divided by its typical slot-to-slot change. |
| inside band | Share of measured values inside the 10% to 90% range (about 0.8 is right). |

### Per-series table

| Column | Meaning |
|---|---|
| forecasts checked | Predictions of this series that have been scored. |
| average miss | Average distance between forecast and measured value. |
| miss if nothing changed | The same, for the "last value stays" guess. |
| skill | 1 - average miss / miss if nothing changed. |
| inside band | Share of measured values inside the forecast range. |

### Backtest
Shown after `sbi backtest` has been run. It replays forecasts from many past moments. Here skill is measured against a harder guess: the average for that weekday and clock time.

## Tab 3: Pipeline

Whether photos get through every stage, and how much of what is extracted reaches the model.

### Tiles

| Tile | Meaning |
|---|---|
| Photos captured | Slots with a photo / slots since the first photo. The bar is green at 98% or more, amber at 90% or more, red below. |
| Numbers per photo | Count of dimensions extracted for this space (all columns of the table). It grows when a new object type or relation is seen for the first time. |
| Usable series | Dimensions with enough history to mean something: values in at least half the slots, not constant, and (for counts) seen in at least 3 slots and 5% of slots. |
| Given to Chronos-2 | Series handed to the model in the latest forecast: the ones it forecasts plus the ones it only reads as inputs. The calendar's 7 columns are included. |
| Time per 15-min run | Typical time one capture-and-read run took over the last 96 runs (24 hours), with the slowest. |

### Photos through each stage
One photo per slot is counted (the one the table uses).

| Stage | Meaning |
|---|---|
| slots since first photo | Every 15-minute step from the first photo to now: the most photos there could be. |
| captured | The camera returned a usable photo. |
| read by detector | YOLO found and boxed the objects (people, chairs, vehicles, ...). |
| read by VLM | The vision-language model returned a valid description (object counts, relations, scene fields). |
| outlined | SAM 2 traced the outline of every boxed object, which gives areas. |
| image vector | DINOv2 produced the whole-picture summary vector. |
| description scores | SigLIP 2 rated the photo against the space's plain-English descriptions. |

### Slots without a photo
Each run of consecutive slots with no photo: first slot, last slot, how many. A photo that is captured but not read yet is not listed here.

### What is extracted and where it goes
One row per family of dimensions, for the latest forecast run.

| Column | Meaning |
|---|---|
| family / what | The kind of number (see "Families of dimensions" below). |
| extracted | Dimensions of this kind in the table. |
| usable | How many pass the usability rules above. |
| input to model (green) | How many the model read as inputs without forecasting them. |
| forecast (blue) | How many the model forecast. |

usable = input to model + forecast, when the `all` input set is in use.

### Every 15-minute slot
A grid with one row per day and one cell per slot (96 per day).

| Colour | Meaning |
|---|---|
| Green | Photo captured and fully read. |
| Dark grey | Photo captured, waiting for the next run to read it. |
| Amber | Photo captured and read by the detector, but the VLM failed, so VLM-based dimensions are empty for that slot. |
| Red | No photo. |

Hover over a cell for its day and time.

### Warnings and errors in the run log
The most recent warning and error lines from `data/logs/tick.log`. An occasional "segmenter unavailable" line means one photo was stored without outlines; the rest of that photo was still read.

## Tab 4: All dimensions

Every dimension extracted for the space, one per row.

- **Family**, **Use**, **Search**: filters. Search matches the readable name or the column name.
- **Plot**: draws any listed numeric dimension over the last 7 days, with its forecast if it has one.

| Column | Meaning |
|---|---|
| dimension | Readable name. |
| column | The exact column name in the table: `family.source.name`. |
| family | The kind of number. |
| now | Value in the latest slot. |
| now (text) | For scene fields that are words (lighting = daylight), the latest value. |
| usual now | Average at this clock time on earlier days. |
| min, max | Lowest and highest value seen so far. |
| +1 h forecast | The middle estimate for one hour ahead, if the series is forecast. |
| coverage | Share of slots that have a value for this dimension. |
| use | **forecast** (blue): the model predicts it. **input to model** (green): the model reads it. **table only** (grey): stored but not given to the model, with the reason. |

Reasons a dimension is "table only":

| Reason | Meaning |
|---|---|
| present in only N slots (flicker) | A count that was above zero in fewer than 3 slots or under 5% of slots: usually a one-off detection. |
| constant | The value never changed, so it carries no information. |
| coverage N% < 50% | Values exist for fewer than half the slots. |
| only N valid slots | Fewer than 8 slots have a value. |
| structural object (building part) | Walls, floor, ceiling, pillars, roof: the VLM's "counts" of these are noise. |

## Families of dimensions

| Family | Example column | Meaning |
|---|---|---|
| scene | `scene.people_yolo` | Totals per photo: people by the detector, people by the VLM, vehicles by the detector, and `count_disagree` (1 when the detector's and the VLM's people counts differ by more than 1). |
| count | `count.yolo.chair` | Number of objects of one type. Source `yolo` = detector boxes with confidence 0.25 or more; `yolo_c05` and `yolo_c10` = all boxes down to 0.05 or 0.10; `vlm` = the VLM's own count; `sam2` = detector count after removing boxes that outline the same object twice. |
| rollup | `rollup.yolo.screen` | Count of a group: the type itself plus every type the vocabulary files under it (for example everything that is a kind of screen). |
| area | `area.sam2.person` | Share of the picture covered by that type, from 0 to 1. `sam2` uses outlines (accurate); `yolo` uses boxes (rougher). |
| cell | `cell.yolo.R5C7.person` | Count of a type in one grid cell. The picture is divided into a grid (cabin 8 x 6, parking 12 x 8); R5C7 = row 5, column 7, counted from the top-left starting at 0. This says where things are. |
| cellarea | `cellarea.sam2.R0C2.monitor` | Share of one grid cell covered by that type's outlines. |
| rel | `rel.vlm.person.sitting_on.chair` | How many times a relation between two object types holds. `vlm` = stated by the VLM; `geometry` = computed from boxes (for example overlaps). |
| img | `img.brightness` | Measured from the pixels: brightness (0 to 255), contrast (spread of brightness), sharpness (edge strength; drops when the picture is blurred). |
| cat | `cat.lighting` | Scene fields the VLM answers in words: lighting, door state, tidiness, visible weather, occupancy level. |
| emb | `emb.dinov2-base.pc1` | From the whole-picture vector. `pc1` to `pc4`: position along the four main ways the scene varies. `drift_prev`: how much the picture changed since the previous slot. `drift_typical`: how far it is from the typical picture. No detector is involved, so these catch changes nothing else names. |
| probe | `probe.siglip2-base-patch16-224.meeting` | How well the photo matches a plain-English description ("a group of people having a meeting"). Higher = closer match; 0 is even odds and most values are negative. Only the movement over time matters, not the level. The descriptions are listed under `probes` in each space's settings file. |
| cal | `cal.hour` | Calendar facts known in advance: hour, minute of day, day of week, weekend, slot of day, holiday, working day. Given to the model for the future as well as the past. |

## Reading it at a glance

- **Is it running?** Header dots say "live".
- **What is happening now?** Tab 1 tiles, against "usual".
- **What will happen?** Tab 1 table and the dashed lines.
- **Can I trust the forecast?** Tab 2: average miss small next to the average value, and "inside the shaded range" near 80%.
- **Is anything being lost?** Tab 3: stages near 100%, few red cells.
- **What exactly do we measure?** Tab 4.

A link can open a space and tab directly: `http://localhost:8502/?space=parking&tab=pipeline` (tab is one of `now`, `accuracy`, `pipeline`, `all`, `guide`).
