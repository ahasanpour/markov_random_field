function MRF_denoisingV2()
%machine learning and pattern recognition bishop chapter 8
% in this version,all pixels around the intended pixel have been considered   
    close all;
    clear all;


    img=imread('/home/ahp/MyMatlabProjects/a1.bmp');
    img_mean=mean(img);
    img_copy=img;
    for i=1:size(img,1)
        for j=1:size(img,2)
            if img(i,j)>img_mean(1,j)
                binary_img(i,j)=-1;
            else
                binary_img(i,j)=1;
            end

        end
    end
    figure;imshow(binary_img)
    img_copy1=binary_img;
    %add noise
     for i=1:((size(binary_img,1)*size(binary_img,2))*20/100)
        a=randi([1,size(binary_img,1)]) ;
        b=randi([1,size(binary_img,2)]) ;
        if binary_img(a,b)==1
            binary_img(a,b)=-1;
        else
            binary_img(a,b)=1;
        end
     end
    figure;imshow(binary_img);
    img_noisy=binary_img;


    %how many percent is really changed
    num=0;
    for i=1:size(img,1)
        for j=1:size(img,2)
            if img_noisy(i,j)~=img_copy1(i,j)
                num=num+1;
            end

        end
    end
    fprintf('percent of noise:%f \n',(num/(size(img_noisy,1)*size(img_noisy,2)))*100);
    %=======================================
    const_list = [0,.1,.02];
    hidden_image = img_noisy;
    total_energy= calculate_total_energy(img_noisy, hidden_image, const_list);
     fprintf('total energy before denoising:%f \n',total_energy);
    
    

    hidden_image = img_noisy;
    energy_this_round = total_energy;
    %fprintf ('% Pixels flipped: %d', percent_pixel_flipped(hidden_image, img_copy))

    for sim_round =1:5
        for i=1 :size(hidden_image,1)-1
            for j=1:size(hidden_image,2)-1
                [hidden_image,should_flip,total_energy] = icm_single_pixel(img_noisy,hidden_image,i,j, total_energy,const_list);
            end
            %print percent_pixel_flipped(hidden_image, lena_arr)
            if (total_energy - energy_this_round) == 0
                %fprintf('Algorithm converged \n');
                continue;
            end
        end
        energy_this_round = total_energy;
        fprintf('Total Energy after denosing in iteration %d is :%f \n',sim_round,total_energy);
        %print "% Pixels flipped:", percent_pixel_flipped(hidden_image, img_arr)
    end
    figure;imshow(hidden_image);
    
    %==========calculate noise rate again
        %how many percent is really changed
    num=0;
    for i=1:size(hidden_image,1)
        for j=1:size(hidden_image,2)
            if hidden_image(i,j)~=img_copy1(i,j)
                num=num+1;
            end

        end
    end
     fprintf('percent of noise after deoising:%f \n',(num/(size(hidden_image,1)*size(hidden_image,2)))*100);
end

function [value]=check_limit(value, limit)

    if value<1
        value=limit;
    end
    if value>=limit
        value=1;
    end
end

function [energy]=add_energy_contribution(visible_arr,hidden_arr, x_val,y_val, const_list)
    h_val = const_list(1);
    beta = const_list(2);
    eta = const_list(3);
    total_pixels =size(hidden_arr,1)*size(hidden_arr,2);
    energy = h_val*hidden_arr(x_val,y_val);
    energy =energy+( -eta*(hidden_arr(x_val,y_val)*visible_arr(x_val,y_val)));
    x_neighbor = [-1,0,1];
    y_neighbor = [-1,0,1];
    for i =1:size( x_neighbor,2)
        
        for j=1:size( y_neighbor,2)
            if not((x_neighbor(i)==0)&&(y_neighbor(j)==0))
                x_n = check_limit(x_val +x_neighbor(i),size(hidden_arr,1));
                y_n = check_limit(y_val +y_neighbor(j), size(hidden_arr,2));
                energy =energy+( -beta*(hidden_arr(x_val,y_val)*hidden_arr(x_n,y_n)));
            end
        end
    end
    energy = energy/total_pixels;
end


function energy1=calculate_total_energy(visible_arr,hidden_arr,const_list)
    energy1 = 0;
    for i=1:size(visible_arr,1)
        
        for j=1:size(visible_arr,2)
            
            energy1 =energy1+ add_energy_contribution(visible_arr,hidden_arr,i,j,const_list);
        end
    end
    energy1;
end
function [hidden_arr,should_flip,total_energy]=icm_single_pixel(visible_arr, hidden_arr, px_x, px_y, total_energy, const_list)
    current_energy = add_energy_contribution(visible_arr, hidden_arr,px_x,px_y, const_list);
    other_energy = total_energy - current_energy;
    %flip the pixel
    new_hidden_arr =hidden_arr;
    if hidden_arr(px_x,px_y)==1
        new_hidden_arr(px_x,px_y)=-1;
    else
        new_hidden_arr(px_x,px_y) = 1;
    end
    flipped_energy = add_energy_contribution(visible_arr, new_hidden_arr,px_x,px_y, const_list);
    %print current_energy, flipped_energy
    if flipped_energy < current_energy
        should_flip = true;
        total_energy = other_energy + flipped_energy;
        hidden_arr = new_hidden_arr;
       
    else
        should_flip = false;
    end
end


    

