% ? means question
% Modified means "I" modified the code
% edited/added lines : %25/12
tic
clear
clc
a=imread('Yale_data_set/1.bmp');a=a(:,(78:320));a=reshape(a,1,59049);
b=imread('Yale_data_set/2.bmp');b=b(:,(78:320));b=reshape(b,1,59049);
c=imread('Yale_data_set/3.bmp');c=c(:,(78:320));c=reshape(c,1,59049);
d=imread('Yale_data_set/4.bmp');d=d(:,(78:320));d=reshape(d,1,59049);
e=imread('Yale_data_set/5.bmp');e=e(:,(78:320));e=reshape(e,1,59049);
f=imread('Yale_data_set/6.bmp');f=f(:,(78:320));f=reshape(f,1,59049);
g=imread('Yale_data_set/7.bmp');g=g(:,(78:320));g=reshape(g,1,59049);
h=imread('Yale_data_set/8.bmp');h=h(:,(78:320));h=reshape(h,1,59049);
i=imread('Yale_data_set/9.bmp');i=i(:,(78:320));i=reshape(i,1,59049);
j=imread('Yale_data_set/10.bmp');j=j(:,(78:320));j=reshape(j,1,59049);
k=imread('Yale_data_set/11.bmp');k=k(:,(78:320));k=reshape(k,1,59049);
l=imread('Yale_data_set/12.bmp');l=l(:,(78:320));l=reshape(l,1,59049);
m=imread('Yale_data_set/13.bmp');m=m(:,(78:320));m=reshape(m,1,59049);
n=imread('Yale_data_set/14.bmp');n=n(:,(78:320));n=reshape(n,1,59049);
o=imread('Yale_data_set/15.bmp');o=o(:,(78:320));o=reshape(o,1,59049);
p=imread('Yale_data_set/16.bmp');p=p(:,(78:320));p=reshape(p,1,59049);
q=imread('Yale_data_set/17.bmp');q=q(:,(78:320));q=reshape(q,1,59049);
r=imread('Yale_data_set/18.bmp');r=r(:,(78:320));r=reshape(r,1,59049);
s=imread('Yale_data_set/19.bmp');s=s(:,(78:320));s=reshape(s,1,59049);
t=imread('Yale_data_set/20.bmp');t=t(:,(78:320));t=reshape(t,1,59049);
features_faces=[a ;b ;c ;d ;e ;f ;g ;h ;i ;j ;k ;l ;m ;n ;o ;p ;q ;r ;s ;t ];
features_faces = im2double(features_faces);
% load('features_faces.mat'); % load the file that has the images.
%In this case load the .mat filed attached above.
% use reshape command to see an image which is stored as a row in the above.
%Postponted
% The code is only skeletal.
% Find psi - mean image
Psi_train = mean(features_faces);%MODIFIED
% Find Phi - modified representation of training images.
% 548 is the total number of training images.
raw_features=features_faces';%MODIFIED
M=size(features_faces,1);%MODIFIED
N2=size(features_faces,2);%MODIFIED
k=5;%Modified
Phi=zeros(N2,M);%MODIFIED to avoid dynamic allocation
for i = 1:M
    Phi(:,i) = raw_features(:,i) - Psi_train';%Modified
end
% Create a matrix from all modified vector images
A = Phi;
% Find covariance matrix using trick above
C = A'*A;
[eig_mat, eig_vals] = eig(C);%eig_vals is the diagonal matrix of C.
%??? why #eig_vals must = M
% Sort eigen vals to get order
eig_vals_vect = diag(eig_vals);
[sorted_eig_vals, eig_indices] = sort(eig_vals_vect,'descend');
sorted_eig_mat = zeros(M);
for i=1:M
    sorted_eig_mat(:,i) = eig_mat(:,eig_indices(i));
end
% Find out Eigen faces % calculate M largest eigenfaces 
Eig_faces = (A*sorted_eig_mat);
size_Eig_faces=size(Eig_faces);%we have got M eigenface each one is N^2*1
% Display an eigenface using the reshape command.
 
% Find out weights for all eigenfaces
% Each column contains weight for corresponding image
W_train = Eig_faces'*Phi;%W_train===col concatenation Large Omiga whose 
%the size of W_train=M(#wieghts)xM(#TrainingFaces) we will reduce it to 
% K(#wieghtsNew)*M(#TrainingFaces).i.e,taking the best K Eig_faces.
%eigenFaces

face_fts = W_train(1:k,:); % features using 250 eigenfaces.%Modified
%Modified
phi_expressed_eigenFaces=zeros(N2,M);
for i=1:M
phi_expressed_eigenFaces(:,i)=Eig_faces(:,(1:k))*face_fts(:,i);
end
%face_fts(:,i) === All waights to produce a cetain training face.
%Eig_faces(:,(1:k))===k effective eigenFaces.
%i ===order of the training face.
% phi_expressed_eigenFaces ===N2xM (logic :D)
% phi_expressed_eigenFaces===conactenation of M colomns of(N2xkxkx1)

%Calculating Omiga of new input


 Im=imread('test.jpg');%25/12
 red_Im=Im(:,:,1);%25/12
green_Im=Im(:,:,2);%25/12
blue_Im=Im(:,:,3);%25/12


threshold=1.2e+3;%25/12
number_of_detections=0;%25/12
% -----------------------------------------
xpixels=size (Im,2);%25/12
ypixels=size (Im,1);%25/12
%  Recognition_matrix=[];%25/12
for i=1:30:xpixels-319%25/12
    for j=1:30:ypixels-242%25/12
       
%  input_image=imread('subject04.sleepy.bmp');%N2x1
input_image = rgb2gray(Im); %25/12
input_image = input_image((j:j+242),(i:i+319));%25/12
input_image=input_image(:,(78:320));input_image=reshape(input_image,1,59049);
input_image = im2double(input_image');
% input_image=ones(N2,1);
phi_input_image=input_image-Psi_train';
face_fts_input=Eig_faces(:,(1:k))'*phi_input_image;
% phi_input_image=== weights to express the mean-adjusted input
% face_fts_input=== weights to express the mean-adjusted input interms of
%"k effective"eigenFaces.
% face_fts_input= k*1 = KxN2xN2x1 = "k effective"eigenFaces*mean-adjusted
% input.

%Calculating euclidean distance
diff = repmat(face_fts_input,1,M) -face_fts ;
square=diff.*diff;
euclidean_distance=sqrt(sum(square,1));
Recognition_IP=find(euclidean_distance<threshold);
% i_j = [i;j];%25/12
% Recognition_matrix=[Recognition_matrix i_j];  25/12
if max(euclidean_distance<threshold)  %for logic 1 %25/12
i=i+77;
 a=i;
 i=j;
 j=a;
red_Im(i:i+242,j)=0;
red_Im(i:i+242,j+242)=0;
red_Im(i,j:j+242)=0;
red_Im(i+242,j:j+242)=0;

green_Im(i:i+242,j)=255;
green_Im(i:i+242,j+242)=255;
green_Im(i,j:j+242)=255;
green_Im(i+242,j:j+242)=255;

blue_Im(i:i+242,j)=0;
blue_Im(i:i+242,j+242)=0;
blue_Im(i,j:j+242)=0;
blue_Im(i+242,j:j+242)=0;


 a=i;
 i=j;
 j=a;
i=i-77;
number_of_detections=number_of_detections+1; %t
end
    end
end
% imshow(reshape(features_faces(Recognition_IP,:),243,243))


Im(:,:,1)=red_Im;%25/12
Im(:,:,2)=green_Im;%25/12
Im(:,:,3)=blue_Im;%25/12

imshow(Im)


toc
number_of_detections %25/12
