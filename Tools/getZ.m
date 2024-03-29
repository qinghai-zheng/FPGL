function [Zmatrix] = getZ(data,marks)
% Function for Z construction of each view
% Notably: this is modified from
% "Large Scale Spectral Clustering with	Landmark-Based Representation," AAAI 2011.

if (~exist('opts','var'))
    opts = [];
end

p = size(marks,1);

r = 20;

nSmp=size(data,1);

% Z construction
D_matrix = L2_distance_construct(data', marks');
[~, idx] = sort(D_matrix, 2);
B_matrix = zeros(nSmp,p);
for ii = 1:nSmp
    id = idx(ii,1:r+1);
    di = D_matrix(ii, id);
    B_matrix(ii,id) = (di(r+1)-di)/(r*di(r+1)-sum(di(1:r))+eps);
end
Zmatrix = B_matrix;


