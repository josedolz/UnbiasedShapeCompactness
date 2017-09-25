% Copyright (c) 2017, Jose Dolz .All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without modification,
% are permitted provided that the following conditions are met:
% 
%     1. Redistributions of source code must retain the above copyright notice,
%        this list of conditions and the following disclaimer.
%     2. Redistributions in binary form must reproduce the above copyright notice,
%        this list of conditions and the following disclaimer in the documentation
%        and/or other materials provided with the distribution.
% 
%     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
%     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
%     OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
%     NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
%     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
%     WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
%     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
%     OTHER DEALINGS IN THE SOFTWARE.
% 
% Jose Dolz. Dec, 2017.
% email: jose.dolz.upv@gmail.com
% LIVIA Department, ETS, Montreal.

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

% N = size(X,1);
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
% M = sparse(T1, T2, 1, N, N);
M = M + M';

%toc





