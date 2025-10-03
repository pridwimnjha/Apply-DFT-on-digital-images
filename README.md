# Apply-DFT-on-digital-images
Great — I went through your assignment document. Since you want a **GitHub README**, I’ll structure it professionally with sections like Overview, Aim, Prerequisites, Procedure, Code, Input/Output, Observations, and Conclusion. Here’s a polished draft tailored for your experiment:

---

# Feature Extraction from Digital Images using Discrete Fourier Transform (DFT)

## 📌 Overview

This project implements **feature extraction from digital images using the Discrete Fourier Transform (DFT)**. The extracted features are then compared using **Euclidean distance** to determine similarity between images. Such methods are useful in **image retrieval systems, fingerprint recognition, and pattern matching**.

---

## 🎯 Aim

* Understand the properties of the Discrete Fourier Transform (DFT).
* Apply DFT on digital images.
* Extract feature vectors (real, imaginary, and phase components).
* Retrieve and compare images from a database using similarity measures.

---

## 📚 Prerequisites

* Basic **MATLAB programming** knowledge.
* Familiarity with **image processing concepts**.
* Dataset of multiple images with varied backgrounds.

---

## ⚙️ Procedure / Algorithm

1. Read the input image.
2. Apply **DFT** using `fft2`.
3. Extract **real, imaginary, magnitude, and phase** components.
4. Distribute values into feature matrices based on **phase angle ranges**.
5. Construct a **feature vector database**.
6. Compare query image features with database features using **Euclidean distance**.
7. Display and analyze the similarity results.

---

## 💻 Code (MATLAB)

```matlab
img = imread("EXP8_IVP.jpg");
a = rgb2gray(img);
img_resized = imresize(a, [512, 512]);
img_resized = double(img_resized);

% Apply DFT
img_dft = fft2(img_resized);
magnitude_img = abs(img_dft);
phase_img = angle(img_dft);
degree = rad2deg(phase_img);

real_img = real(img_dft);
imagimg = imag(img_dft);

% Phase-based segmentation
I1 = zeros(512, 512); R1 = zeros(512, 512);
I2 = zeros(512, 512); R2 = zeros(512, 512);

for i = 1:512
   for j = 1:512
       if (degree(i, j) >= 0 && degree(i, j) < 90)
           I1(i, j) = imagimg(i, j);
           R1(i, j) = real_img(i, j);
       elseif (degree(i, j) >= 90 && degree(i, j) < 180)
           I2(i, j) = imagimg(i, j);
           R2(i, j) = real_img(i, j);
       end
   end
end

databaseFeature = [I1(:); R1(:); I2(:); R2(:)]';

% Query Image
query_img = imread("EXP6_IMG.jpg");
query_gray = rgb2gray(query_img);
query_resized = imresize(query_gray, [512, 512]);
query_resized = double(query_resized);
query_dft = fft2(query_resized);

query_phase = angle(query_dft);
query_degree = rad2deg(query_phase);
query_real = real(query_dft);
query_imag = imag(query_dft);

I1_query = zeros(512, 512); R1_query = zeros(512, 512);
I2_query = zeros(512, 512); R2_query = zeros(512, 512);

for i = 1:512
   for j = 1:512
       if (query_degree(i, j) >= 0 && query_degree(i, j) < 90)
           I1_query(i, j) = query_imag(i, j);
           R1_query(i, j) = query_real(i, j);
       elseif (query_degree(i, j) >= 90 && query_degree(i, j) < 180)
           I2_query(i, j) = query_imag(i, j);
           R2_query(i, j) = query_real(i, j);
       end
   end
end

queryFeature = [I1_query(:); R1_query(:); I2_query(:); R2_query(:)]';

distance = pdist2(queryFeature, databaseFeature, 'euclidean');
fprintf('Euclidean Distance: %.2f\n', distance);
```

---

## 🖼️ Input / Output

* **Input:**

  * Original Image (`EXP8_IVP.jpg`)
    
  * Query Image (`EXP6_IMG.jpg`)

* **Output:**

  * Fourier transform representation.
  * Extracted features.
  * Euclidean distance score for similarity.

<img width="632" height="357" alt="image" src="https://github.com/user-attachments/assets/33e8e9f7-10d8-4bf0-867e-b8fa18606a21" />

---

## 🔍 Observations & Learning

* **Fourier Transform:** Allows analysis in the frequency domain.
* **Phase Segmentation:** Helps capture structural details.
* **Similarity Metric:** Euclidean distance indicates closeness between images.
* **Optimization:** Loops can be vectorized for efficiency.
* **DFT Visualization:** Should use `imshow(log(1+abs(img_dft)), [])` instead of `imshow(img_dft)`.

---

## ✅ Conclusion

This experiment demonstrates **feature extraction using DFT** and similarity measurement with **Euclidean distance**. It highlights the importance of **frequency domain analysis** in image processing and its applications in **image retrieval systems and biometric recognition**.

---

## ❓ Questions of Curiosity

**Q1:** What are other similarity measures besides Euclidean distance?

* Manhattan Distance
* Chebyshev Distance
* Minkowski Distance

**Q2:** How does fingerprint recognition matching work?

* Acquisition → Preprocessing → Feature Extraction (minutiae points) → Matching using similarity scores.


