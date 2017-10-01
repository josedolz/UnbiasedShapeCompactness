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

function [Seg,Seg0, res] = compactnessSegProbMap(Image, probMap, P)

smallEps = 1e-6;

[H,W] = size(Image);
N = W*H;

X = Image(:);

% Compute Laplacian
tic
W = computeWeights(Image, P.Kernel, P.sigma, P.eps);
toc
L = spdiags(sum(W,2),0,N,N) - W;

% Compute unary from prob maps
Priors = probMap;

% Unary potentials are -log posteriors
U0 = zeros(2,N);
U0(1,:) = -log(smallEps  + (1-Priors(:)));
U0(2,:) = -log(smallEps  +  Priors(:));

p = log(probMap(:));

V0 = U0;
V = V0;

hbk = BK_Create(N,nnz(P.Kernel)*N);

BK_SetNeighbors(hbk,W);
BK_SetUnary(hbk,U0);
E = BK_Minimize(hbk);
disp(num2str(E))
y0 = double(BK_GetLabeling(hbk)) - 1;

Seg0 = reshape(y0, size(Image));

y = y0;

c = sum(y0);
o = ones(N,1);
u = zeros(N,1);
v = 0;
tt = y'*L*y;

cost1Prev = 0;
for i = 1:P.maxLoops

    %%%%%%%%%%%%%%%%%%%%%%%%%% UPDATE z %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    alpha = (P.lambda/c)*tt;

    if P.solvePCG
        [temp,~] = pcg(alpha*L + P.mu1*speye(N), P.mu1*(y+u) + P.mu2*(c+v));
    else
        a = (alpha*L + P.mu1*speye(N));
        b = (P.mu1*(y+u) + P.mu2*(c+v));
        temp = a \ b;
    end

    const = (1/P.mu1)*(1/P.mu2 + N/P.mu1).^(-1);
    z = temp - const*sum(temp)*o;
    % SANITY CHECK
    %z2 = (alpha*L + P.mu1*speye(N) + P.mu2*ones(N,N)colormap(jet))\(P.mu1*(y+u) + P.mu2*(c+v)*o);
    %diff = norm(z-z2)

    %%%%%%%%%%%%%%%%%%%%%%%%%% UPDATE c %%%%%%%%%%%%%%%%%%%%%%%%%
    rr = z'*L*z;
    beta = 0.5*P.lambda*tt*rr;

    qq = sum(z) - v;

    R = roots([1 -qq 0 -beta/P.mu2]);
    R = R(imag(R)==0);

    if isempty(R)
        disp('No roots found... error');
        i = 1;
        P.lambda = P.lambda/10;
        %break;
    end

    c = max(R);

    %%%%%%%%%%%%%%%%%%%%%%%%%% UPDATE y %%%%%%%%%%%%%%%%%%%%%%%%%
    gamma = 0.5*(P.lambda/c)*rr;
    V(2,:) = (p + P.mu1*(u-z+0.5))'/(gamma + P.lambda0);

    BK_SetUnary(hbk,V);
    E = BK_Minimize(hbk);
    y = double(BK_GetLabeling(hbk)) - 1;
    tt = y'*L*y;

    %%%%%%%%%%%%%%%%%%%%%%%%%% UPDATE Lagrangian mult. %%%%%%%%%%%%%%%%%%%%%%%%%
    u = u + (y-z);
    v = v + (c-sum(z));

    Seg = reshape(y, size(Image));

    if P.dispSeg
        figure(100),
        subplot(1,5,1), imshow(Image,[]);
        subplot(1,5,2), imshow(P.GroundTruth,[]);
        subplot(1,5,3), imshow(probMap,[]);
        subplot(1,5,4), imshow(Seg0,[]);
        subplot(1,5,5), imshow(Seg,[]);
        %
        %     %figure(2), hist(z,100);
        drawnow;
    end

    cost1 = p'*y;
    cost2 = P.lambda*(tt^2/sum(y));
    cost3 = norm(y-z);
    cost4 = norm(c - sum(z));


    if P.dispCost
        fprintf('%d : %f %f %f %f %f %f\n', i, cost1, cost2, cost3, cost4, P.mu1, P.mu2);
    end


    reg = regionprops(Seg,'BoundingBox','Area');
    if(isempty(reg))
       res = -1;
       disp('Trying with decreased lambda...')
       return
    else
        if (reg(1).Area == N)
           res = -2;
           disp('Full slide');
           return
        end
    end

    P.mu1 = P.mu1*P.mu1Fact;
    P.mu2 = P.mu2*P.mu2Fact;

    if cost1Prev == cost1
       res = 0;
       return
    else
       cost1Prev = cost1;
    end


end
res = 0;
BK_Delete(hbk);

Seg = reshape(y, size(Image));

