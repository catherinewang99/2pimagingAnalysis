figure()
for i=1:10
    subplot(2,5,i)
    for j=1:size(obj.roi_plane{1,6}.F,2)
        hold on
        %plot(obj.roi_plane{1,6}.Fraw{1,j}(i,:)-obj.roi_plane{1,6}.F{1,j}(i,:))
        plot(obj.roi_plane{1,6}.F{1,j}(i,:))
    end
    xlim([0 48])
end
