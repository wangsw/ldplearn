
# ldp-learn


## A library for local differential private data analyses.


Local differential privacy (LDP)  is the state-of-the-art notion for data privacy, emerges as a remedy for data privacy issue in this big data era.  LDP is fexlexible, lightweight and easy to implement for production envrionments. One complelling property of LDP is that the data owner could have full control of their data without trust of any other parties. 

This reposity provides serveral LDP mechanisms intended for user/client data analyses, including distribution estimation and mean estimation (for now) on categorical data, set-valued data and numerical data.


## Supported mechanism
1. binary randomized response (brr)[3] for categorical data distribution estimation.
2. multinomial randomzed response (mrr)[3] for categorical data distribution estimation.
3. k-subset mechanism (k-subset)[3] for categorical data distribution estimation.


4. binary randomized repsonse for set-valued data (brrset, RAPPOR) [1] distribution estimation
5. privset mechanism (privset)[2] for set-valued data distribution estimation.





[1] Erlingsson, Úlfar, Vasyl Pihur, and Aleksandra Korolova. "Rappor: Randomized aggregatable privacy-preserving ordinal response." Proceedings of the 2014 ACM SIGSAC conference on computer and communications security. ACM, 2014.

[2] Wang Shaowei, Huang Liusheng, Nie Yiwen, Wang Pengzhan, Xu Hongli, Yang Wei. "PrivSet: Set-valued Data Analyses with Local Differential Privacy." The 37th Annual IEEE International Conference on Computer Communications (INFOCOM). IEEE, 2018. 

[3] Wang Shaowei, Huang Liusheng, Wang Pengzhan, Nie Yiwen, Xu Hongli, Yang Wei, Qiao, Chunming. "Mutual Information Optimally Local Private Discrete Distribution Estimation." arXiv preprint arXiv:1607.08025.
