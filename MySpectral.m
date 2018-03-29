function [ClusterIm, CCIm] = MySpectral(Im, ImType, NumClusts)
% Spectral clustering algorithm for RGB / Hyperspectral Image Clustering
% 
% Syntax: [ClusterIm, CCIm] = MyClust(Im, ImType, NumClusts)
% 
% Inputs:
%     Im - [input image]
%     ImType - [string argument] - image type - ('RGB','Hyper')
%     NumClusts - [positive integer] - number of clusters
%     
% Outputs:
%     ClusterIm - [double Mat] - labeled image with dimension row x col
%     CCIm - [double Mat] - tensor of binary image representing connected
%     components in ClusterIm (for RGB images ONLY)
% 
% Author: Pallavi A Raiturkar
% University of Florida, Dept of Computer and Information Science and Engineering
%%

if (strcmp(ImType,'RGB'))
    
    %resize image to speed up computation
    Img = imresize(Im,[50 50]);
    %reshape into features
    [nrows, ncols, d] = size(Img);
    I = reshape(Img, 1, nrows * ncols, []);
    if (d >= 2)
    I = (squeeze(I))';
    end
    I = double(I);
    %normalize features
    I = normalizeData(I);
%%
    %compute similarity graph
%     SimGraph = SimGraph_Epsilon(I, 0.3);
    epsilon = 0.3;
    n = size(I,2);
    
    
    indi = [];
    indj = [];
    inds = [];
    
    for ii = 1:n
    % Compute i-th column of distance matrix
    dist = sqrt(sum((repmat(I(:, ii), 1, n) - I) .^ 2, 1));
    % Find distances smaller than epsilon (unweighted)
    dist = (dist < epsilon);
    
    % Now save the indices and values for the adjacency matrix
    lastind  = size(indi, 2);
    count    = nnz(dist);
    [~, col] = find(dist);
    
    indi(1, lastind+1:lastind+count) = ii;
    indj(1, lastind+1:lastind+count) = col;
    inds(1, lastind+1:lastind+count) = 1;
    end

    % Create adjacency matrix for similarity graph
    W = sparse(indi, indj, inds, n, n);

    clear indi indj inds dist lastind count col v;
%%    
    % calculate degree matrix
    degs = sum(W, 2);
    D    = sparse(1:size(W, 1), 1:size(W, 2), degs);

    % compute unnormalized Laplacian
    L = D - W;

    % compute normalized Laplacian if needed
            % avoid dividing by zero
            degs(degs == 0) = eps;
            % calculate inverse of D
            D = spdiags(1./degs, 0, size(D, 1), size(D, 2));

            % calculate normalized Laplacian
            L = D * L;
        
    % compute the eigenvectors corresponding to the k smallest
    % eigenvalues
    diff   = eps;
    [U, ~] = eigs(L, NumClusts, diff);

    % now use the k-means algorithm to cluster U row-wise
    % C will be a n-by-1 matrix containing the cluster number for
    % each data point
    C = kmeans(U, NumClusts, 'start', 'cluster', ...
                     'EmptyAction', 'singleton');

    % now convert C to a n-by-k matrix containing the k indicator
    % vectors as columns
    C = sparse(1:size(D, 1), C, 1);
%%
    %convert cluster vector
    
    if size(C, 2) > 1
        indMatrix = zeros(size(C, 1), 1);
        for ii = 1:size(C, 2)
            indMatrix(C(:, ii) == 1) = ii;
        end
    else
        indMatrix = sparse(1:size(C, 1), C, 1);
    end
    
    ClusterIm = reshape(indMatrix, ncols, nrows);
    
    % find connected components in each cluster
    CCIm = zeros(NumClusts, nrows * ncols); 
    
    for i = 1:NumClusts
        index = indMatrix == i;
        CCIm(i, :) = index * 1;
    end
    
elseif (strcmp(ImType, 'Hyper'))
    
    %resize image to speed up computation
    Img = imresize(Im,[50 50]);
    %reshape into features
    [nrows, ncols, d] = size(Img);
    I = reshape(Img, 1, nrows * ncols, []);
    if (d >= 2)
    I = (squeeze(I))';
    end
    I = double(I);
    %normalize features
    I = normalizeData(I);
%%
    %compute similarity graph
%     SimGraph = SimGraph_Epsilon(I, 0.3);
    epsilon = 0.3;
    n = size(I,2);
    
    
    indi = [];
    indj = [];
    inds = [];
    
    for ii = 1:n
    % Compute i-th column of distance matrix
    dist = sqrt(sum((repmat(I(:, ii), 1, n) - I) .^ 2, 1));
    % Find distances smaller than epsilon (unweighted)
    dist = (dist < epsilon);
    
    % Now save the indices and values for the adjacency matrix
    lastind  = size(indi, 2);
    count    = nnz(dist);
    [~, col] = find(dist);
    
    indi(1, lastind+1:lastind+count) = ii;
    indj(1, lastind+1:lastind+count) = col;
    inds(1, lastind+1:lastind+count) = 1;
    end

    % Create adjacency matrix for similarity graph
    W = sparse(indi, indj, inds, n, n);

    clear indi indj inds dist lastind count col v;
%%    
    % calculate degree matrix
    degs = sum(W, 2);
    D    = sparse(1:size(W, 1), 1:size(W, 2), degs);

    % compute unnormalized Laplacian
    L = D - W;

    % compute normalized Laplacian if needed
            % avoid dividing by zero
            degs(degs == 0) = eps;
            % calculate inverse of D
            D = spdiags(1./degs, 0, size(D, 1), size(D, 2));

            % calculate normalized Laplacian
            L = D * L;
        
    % compute the eigenvectors corresponding to the k smallest
    % eigenvalues
    diff   = eps;
    [U, ~] = eigs(L, NumClusts, diff);

    % now use the k-means algorithm to cluster U row-wise
    % C will be a n-by-1 matrix containing the cluster number for
    % each data point
    C = kmeans(U, NumClusts, 'start', 'cluster', ...
                     'EmptyAction', 'singleton');

    % now convert C to a n-by-k matrix containing the k indicator
    % vectors as columns
    C = sparse(1:size(D, 1), C, 1);
%%
    %convert cluster vector
    
    if size(C, 2) > 1
        indMatrix = zeros(size(C, 1), 1);
        for ii = 1:size(C, 2)
            indMatrix(C(:, ii) == 1) = ii;
        end
    else
        indMatrix = sparse(1:size(C, 1), C, 1);
    end
    
    ClusterIm = reshape(indMatrix, ncols, nrows);
    
    CCIm = []; % connected components are not required for hyperspectral images
end




end