function drawResults(img, GT, CNN_Seg, GCs_Seg, ADMM_Seg)

subplot(1,4,1)
imshow(img,[])
hold on
title ('Ground truth')

contour = bwboundaries(im2bw(double(GT)));
for k = 1:length(contour)
        boundary = contour{k};
        plot(boundary(:,2), boundary(:,1), 'Color','g' ,'LineWidth', 2)
end
   
subplot(1,4,2)
imshow(img,[])
hold on
title ('Seg (CNN)')
contour = bwboundaries(im2bw(double(CNN_Seg)));
for k = 1:length(contour)
        boundary = contour{k};
        plot(boundary(:,2), boundary(:,1), 'Color','g' ,'LineWidth', 2)
end

subplot(1,4,3)
imshow(img,[])
hold on
title ('Seg (GCs)')
contour = bwboundaries(im2bw(double(GCs_Seg)));
for k = 1:length(contour)
        boundary = contour{k};
        plot(boundary(:,2), boundary(:,1), 'Color','g' ,'LineWidth', 2)
end


subplot(1,4,4)
imshow(img,[])
hold on
title ('Seg (ADMM)')
contour = bwboundaries(im2bw(double(ADMM_Seg)));
for k = 1:length(contour)
        boundary = contour{k};
        plot(boundary(:,2), boundary(:,1), 'Color','g' ,'LineWidth', 2)
end