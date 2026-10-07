# ApplicationImageColorQuantization

## Introduction
Color quantization reduces the number of distinct colors in an image so that the result is visually similar to the original but cheaper to store. A pixel is an RGB triplet, i.e. a point in a 3-D color space; a 24-bit image may use up to 16.7 million colors. If the image is re-expressed with only *k* representative colors (a palette), each pixel needs just ⌈log2 k⌉ bits plus a small shared palette. K-means clustering (MacQueen, 1967; Lloyd, 1982) is a natural fit: it partitions the pixels into *k* clusters of similar color and uses each cluster's mean as the palette color.

## 1. Algorithm
1. **Flatten**: reshape the H×W×3 image into N = H·W points in RGB space.
2. **Initialize**: choose *k* random pixels as initial centroids.
3. **Assignment step**: assign every pixel to the nearest centroid by squared Euclidean distance, `argmin_j ||x - c_j||²`.
4. **Update step**: recompute each centroid as the mean of the pixels assigned to it (an empty cluster keeps its centroid).
5. **Repeat** steps 3–4 until the centroids move less than a tolerance or a maximum iteration count is reached. K-means monotonically decreases the within-cluster sum of squares, so it converges to a local minimum (Lloyd, 1982); the result depends on initialization.
6. **Reconstruct**: replace each pixel by its centroid (rounded to 0–255). Store the *k* centroids as a palette and one index per pixel (indexed PNG).

Cost per iteration is O(N·k·3); storage falls from 24 bits per pixel to ⌈log2 k⌉ bits per pixel plus 3k bytes of palette.

## 2. Code
See [`quantize.py`](quantize.py) (requires `numpy` and `Pillow`: `pip install -r requirements.txt`).

```
python quantize.py [input_image] -k 16 -o quantized.png
```
Without an input image a synthetic gradient demo image is generated. The tool prints the number of colors, MSE/PSNR and file size.

## 3. Example output
Running `python quantize.py -k 16` on the 256×256 demo image:

```
colors in original : 52688
colors (k)         : 16
MSE / PSNR         : 193.97 / 25.25 dB
raw size           : 196608 bytes (24 bpp)
quantized file     : 1944 bytes -> /tmp/q16.png
```
The 52k-color image is reduced to 16 colors; the raw 196,608-byte RGB data shrinks to a palette PNG of about 2 KB at the PSNR shown (values above ~30 dB are usually visually close; lower k trades quality for size, e.g. k=8 gives ≈22 dB on this gradient-heavy image). Try it on any photograph with different values of k.

## 4. Benefits and limitations
**Benefits**
- Large size reduction (e.g. 24 → 4 bits per pixel for k=16) and a simple, easily explained algorithm.
- The palette adapts to the image's own color distribution, giving lower error than a fixed palette.
- Output is a standard indexed image readable everywhere; useful for web graphics, posters, and segmentation.
- Quality/size is tunable through *k*.

**Limitations**
- Lossy: smooth gradients show banding and fine detail/color is lost (dithering can mitigate).
- Sensitive to initialization and may reach only a local optimum (k-means++ by Arthur & Vassilvitskii, 2007, helps).
- *k* must be chosen in advance; Euclidean RGB distance is not perceptually uniform (CIELAB would be better).
- Slow on large images (O(N·k) per iteration), and small but visually important colors may be absorbed into larger clusters.
- It only exploits color redundancy, not spatial redundancy, so transform codecs such as JPEG are usually more efficient for photos.

## References
- Lloyd, S. (1982). Least squares quantization in PCM. *IEEE Trans. Information Theory*, 28(2), 129–137.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proc. 5th Berkeley Symposium*, 281–297.
- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *SODA '07*, 1027–1035.
- Heckbert, P. (1982). Color image quantization for frame buffer display. *SIGGRAPH '82*, 297–307.
