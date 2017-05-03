% NOTE: Kernel is a matrix of binary values representing the neighbors
%
% Example: For a 3x3 neighborhood
%
% Kernel =
%   1     1     1
%   1     0     1
%   1     1     1
%
function M = computeWeights(Image, Kernel, sigma, eps)

[W,H] = size(Image);
N = W*H;
X = Image(:);

[KW,KH] = size(Kernel);
K = nnz(Kernel);

N = size(X,1);
A = padarray(reshape(1:N,W,H),[ceil(0.5*(KW-1)) ceil(0.5*(KH-1))]);
Neigh = zeros(N,K);

k = 1;
for i=1:KW
    for j=1:KH  
        if Kernel(i,j) == 0
            continue;
        end
        
       T = A(i:i+W-1, j:j+H-1);       
       Neigh(:,k) = T(:);
       k = k+1;
    end
end

T1 = repmat((1:N)', K, 1);
T2 = Neigh(:);
Z = (T1 > T2);
T1(Z) = [];
T2(Z) = [];

M = sparse(T1, T2, (1-eps)*exp(-sigma*(X(T1,1)-X(T2,1)).^2) + eps, N, N);
%M = sparse(T1, T2, 1, N, N);
M = M + M';

%toc





