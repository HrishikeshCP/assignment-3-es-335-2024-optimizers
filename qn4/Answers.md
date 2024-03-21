# Q4
## Part A
Architecture of MLP : 
<img src='/Users/aravindkrishna/assignment-3-es-335-2024-optimizers/qn4/MLP_arch.png'>
MLP Stats :
<img src='/Users/aravindkrishna/assignment-3-es-335-2024-optimizers/qn4/MLP_stats.png'>
Logistic Regression Stats :
<img src='/Users/aravindkrishna/assignment-3-es-335-2024-optimizers/qn4/LR_stats.png'>
Random Forest Stats:
<img src='/Users/aravindkrishna/assignment-3-es-335-2024-optimizers/qn4/RF_stats.png'>

As seen from the images <b>qualitatively</b> the reconstructed image become better as we increase the number of RFF features. When the number of RFF features is 20,000 then the reconstructed image is very similar to the Original Image.



<b> Quantitatively</b> we can observe that the <b>RMSE value decreases</b> as we increase the number of RFF features while the <b>PSNR increases</b> which is as expected.

## Part C
t-SNE of Untrained:
<img src='/Users/aravindkrishna/assignment-3-es-335-2024-optimizers/qn4/t-SNE of the 64-neuron layer output (Untrained Model).png'>
t-SNE of Trained:
<img src='/Users/aravindkrishna/assignment-3-es-335-2024-optimizers/qn4/t-SNE of the 64-neuron layer output (Trained Model).png'>
t-SNE of Fashion MNIST 
<img src='/Users/aravindkrishna/assignment-3-es-335-2024-optimizers/qn4/t-SNE of the 64-neuron layer output (Fashion Trained Model).png'>

We can observe that the <b>RMSE value decreases</b> with the increase in percentage of data missing data. 
The PSNR decreases with the increase in percentage of missing data which is as expected. The reconstruction till 50% of the data missing is fairly good which after 50 the quality decreases significantly. 
The slope of RMSE is very close to 0 till 50% missing data while the it +ve'ly increase drastically after 50& missing data. Same is for PSNR, the difference being that the value of -ve slope increases.