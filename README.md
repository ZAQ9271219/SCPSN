# SCPSN-Spectral-Clustering-based-Pyramid-Super-resolution-Network-for-Hyperspectral-Images
This is the code for the ACM MM 2024 paper :SCPSN: Spectral Clustering-based Pyramid Super-resolution  Network for Hyperspectral Images.
We will make it public after publication
<h1>Request</h1>
<code>pytorch 1.13.1</code><br/>
<code>scikit-learn 1.3.2</code>
<h1>Usage</h1>
This is a setting examples as a scale factor of ×4
<code>model = main(in_channels=191, num_clusters=16, )</code><br/>
<code>out, out1, out2= model(MS_image)<br/>
loss = criterion(outputs, to_variable(reference)) 
loss1 = criterion(out1, to_variable(downX2(reference)))<br/>
loss2 = criterion(out2, to_variable(downX2(downX2(reference))))<br/>
SAM_loss = SAM(outputs, to_variable(reference))<br/>
loss = loss + SAM_loss / (SAM_loss / loss).detach() + loss1 / (loss1 / loss).detach() + loss2 / (loss2 / loss).detach() <br/>
</code>
