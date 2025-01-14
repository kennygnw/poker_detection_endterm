# 嵌入式影像處理

## 撲克牌分辨

### 授課教授：陳朝烈老師

### 組員：曾鈺恩，朱泂樺，黃湟喜，吳永保

## 規格

### 功能
- Unit Test: 輸入德州撲克影片，分類畫面上的影片有哪幾張卡
### 軟硬體規格
- 軟體：
    - Python 3.4 以上
    - OpenCV套件
- 硬體：
    - CPU運算能力 (不需要使用GPU運算)
    - 攝影機 (或影片檔)
    - 30FPS/720p
    - FOV 約 50度
    - 擺設高度約95公分 
    - 桌面布色為長方形深綠色或深藍色，可3人玩的空間&大小 (照攝直徑約100cm
- 效能
    - FPS: 目標達到15FPS
- 限制
    - 桌上畫面，非第一視角
    - 著光相對均勻，桌面乾淨
    - 撲克牌種類為邊角使用標準牌型
- 介面
    - 輸入
        - 影片檔(.mkv或.mp4)
        - USB攝影機傳輸
    - 輸出
        - OS裡的Videostream畫面(720p視窗)
## Breakdown
![alt text](/readme_images/breakdown_picture.png)

## Flowchart
![alt text](/readme_images/flowchart_picture.png)

## API
- 二值化 Adaptive Threshold
    - 輸入：灰階array U8 [1280,720]
    - 輸出：二值化array U8 [1280,720]
    - 功能：在kernel裡計算平均值，如果pixel與平均值超過閥值，該pixel為255
    - 參數：閥值，kernel
    - Unit Test

![alt text](/readme_images/adaptivethres_usecase.png)

- 找輪廓
    - 輸入：二值化array U8 [1280,720]
    - 輸出：輪廓array (I32)
    - 參數：Hierarchy：只取外部
    - Unit Test

![alt text](/readme_images/getcontour_usecase.png)

- 判斷面積
    - 輸入：輪廓array (I32)
    - 輸出：篩選後輪廓array (I32)
    - 功能：使用輪廓計算面積，如果判斷超過允許範圍跳過該輪廓組
    - 參數：
        - 最小面積：(螢幕寬度*10%) * (螢幕高度*5%)
        - 最大面積：(螢幕寬度*30%) * (螢幕高度*20%)
    - Unit Test

![alt text](/readme_images/countarea_usecase.png)

- 取輪廓邊角
    - 輸入：二值化輸出 U8 array[1280,720], 篩選後輪廓array I32
    - 輸出：邊角array組 float32 [13, 4, 2]
    - 功能：判斷輪廓裡的邊角，如果判斷4個邊角繼續流程
    - Unit Test

![alt text](/readme_images/getcorner_usecase.png)

- 透視變換
    - 輸入：邊角array組 float32 [13, 4, 2]
    - 輸出：拉正後卡array U8 [h,w]
    - 參數：透視變換高度，長度 h, w：300, 200
    - Unit Test

![alt text](/readme_images/warppers_usecase.png)

- 分割花色、數字
    - 輸入：邊角array u8 [h*m, w*n]
    - 輸出：判斷出數字跟花色單獨區域 array u8 [y, x]
    - 功能：抓出單獨數字及花色面輪廓，利用boundingRect w*h，判斷面積區域分割花色及數字
    - Unit Test

![alt text](/readme_images/getnum_pattern_usecase.png)

- 與對照組進行XOR運算
    - 輸入：偵測區域圖片 array u8 [y, x]
    - 輸出：判斷花色及數字差異(取最小) (int)
    - 功能：Xor當下擷取到的圖片矩陣與對照組圖片矩陣做比對
    - 參數：對照組圖片路徑 (Path() 物件)
    - Unit Test

![alt text](readme_images/xor_calculations.jpg)

*輸入圖片*

![alt text](readme_images/xor_calculations_result.png)

*輸出結果，再取最低值*

### 對照組

<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;">
  <img src="number/A.jpg" alt="Image 1" style="width: 5%;">
  <img src="number/2.jpg" alt="Image 2" style="width: 5%;">
  <img src="number/3.jpg" alt="Image 3" style="width: 5%;">
  <img src="number/4.jpg" alt="Image 4" style="width: 5%;">
  <img src="number/5.jpg" alt="Image 5" style="width: 5%;">
  <img src="number/6.jpg" alt="Image 6" style="width: 5%;">
  <img src="number/7.jpg" alt="Image 6" style="width: 5%;">
  <img src="number/8.jpg" alt="Image 6" style="width: 5%;">
  <img src="number/9.jpg" alt="Image 6" style="width: 5%;">
  <img src="number/10.jpg" alt="Image 6" style="width: 5%;">
  <img src="number/J.jpg" alt="Image 6" style="width: 5%;">
  <img src="number/Q.jpg" alt="Image 6" style="width: 5%;">
  <img src="number/K.jpg" alt="Image 6" style="width: 5%;">
</div>

<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;">
  <img src="suit/spade.jpg" alt="Image 1" style="width: 10%;">
  <img src="suit/club.jpg" alt="Image 2" style="width: 10%;">
  <img src="suit/heart.jpg" alt="Image 3" style="width: 10%;">
  <img src="suit/diamond.jpg" alt="Image 4" style="width: 10%;">
</div>

*使用標準撲克牌的dataset*

- 撲克牌命名
    - 輸入：原frame u8 array[1280,720]
    - 輸出：追蹤dictionary, 命名後 frame u8 array[1280,720]
    - 功能：命名偵測到的撲克牌，因為偵測到會有少次誤判，定一個更新標籤計數器的參數，統計每frame偵測到在同一個位置出現的花色與數字。計數器到將在同一個位置的所偵測到的花色與數字的mode取下。
    - 參數：偵測frame計數器threshold (uint8)
    - Unit Test

![alt text](readme_images/tracker_dict.jpg)

*輸入畫面上偵測到的花色及數字資訊*

![alt text](readme_images/tracker_dict_result.png)

*計數器到達前累積的資訊，dictionary key為["x","y","結果"] value為偵測到次數*
## Input Video

[https://youtu.be/7EZrWosCTPI](https://youtu.be/7EZrWosCTPI)

[https://youtu.be/HG8ZHYpgHAE](https://youtu.be/HG8ZHYpgHAE)

## Output Result

[https://youtu.be/wDtiPlYHCXA](https://youtu.be/wDtiPlYHCXA)

[https://youtu.be/B9E0-0nMQy8](https://youtu.be/B9E0-0nMQy8)

[https://youtu.be/QiSMg0GN4NM](https://youtu.be/QiSMg0GN4NM)

## 工作分配

### 曾鈺恩：花色與數字資料擷取、比對、副程式協調

### 吳永保：輪廓處理、主程式整合、卡片追蹤

### 朱泂樺：前處理、資料與readme統整

### 黃湟喜：二值化處理、資料與readme統整