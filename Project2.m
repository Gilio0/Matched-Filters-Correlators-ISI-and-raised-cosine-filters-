close all;
clear ;
clc;
 
pulse_shape = [5 4 3 2 1]/sqrt(55);
number_of_bits=10;
array_of_10_random_bits = randi([0, 1], 1, number_of_bits);
 
mapped_array_for_10_bits = (2*array_of_10_random_bits)-1; %mapped 0 to -1 & mapped 1 to 1
 
sampled_array = upsample(mapped_array_for_10_bits,5);
 
Transmitted_signal = conv(sampled_array,pulse_shape);
Transmitted_signal = Transmitted_signal(1:50);
 
matched_filter = fliplr(pulse_shape);
 
unity_filter = [1 1 1 1 1]/sqrt(5);
 
matched_filter_output = conv(Transmitted_signal,matched_filter);
matched_filter_output = matched_filter_output(1:50);
 
unity_filter_output = conv(Transmitted_signal,unity_filter);
unity_filter_output = unity_filter_output(1:50);
 
figure(1);
 
subplot(2,1,1);
plot(matched_filter_output,'r', 'LineWidth', 2);
hold on;
stem(matched_filter_output);
legend('Matched filter output','Matched filter output sampled');
title('Matched filter output');
ylabel('Amplitude');
xlabel('Number of sample');
grid on;
hold off;
 
subplot(2,1,2);
plot(unity_filter_output,'b', 'LineWidth', 2);
hold on;
stem(unity_filter_output);
legend('Unity filter output','Unity filter output sampled');
title('Unity filter output');
ylabel('Amplitude');
xlabel('Number of sample');
grid on;
hold off;
 
Correlator_output = zeros(1, number_of_bits*5);
 
for i=1:5:50
    Correlator_output(i+4) = sum(Transmitted_signal(i:i+4).*pulse_shape);
end
 
 
 
figure(2);
plot(matched_filter_output,'r', 'LineWidth', 2);
hold on;
stem(matched_filter_output);
legend('Matched filter output','Matched filter output sampled');
title('Matched filter output');
ylabel('Amplitude');
xlabel('Number of sample');
grid on;
plot(Correlator_output,'b', 'LineWidth', 2);
hold on;
stem(Correlator_output);
legend('Matched filter output','Matched filter output sampled','Unity filter output','Unity filter output sampled');
title('Unity filter output');
ylabel('Amplitude');
xlabel('Number of sample');
grid on;
hold off;
 
number_of_bits=10000;
array_of_10000_random_bits = randi([0, 1], 1, number_of_bits);
mapped_array_for_10000_bits = (2*array_of_10000_random_bits)-1; %mapped 0 to -1 & mapped 1 to 1
sampled_array_for_10000_bits = upsample(mapped_array_for_10000_bits,5);
Transmitted_signal_for_10000_bits = conv(sampled_array_for_10000_bits,pulse_shape);
size_of_transmitted_signal=size(Transmitted_signal_for_10000_bits,2);
filter_2=[5,5,5,5,5];
filter_2=[5,5,5,5,5]/sqrt(sum(filter_2(:).^2));
Eb=sum(pulse_shape(:).^2); % computing the energy of the bit
SNR_db=-2:1:5;   % SNR=Eb/No  where No is the noise power
SNR_linear=zeros(1,size(SNR_db,2));
BER_matched_filter=zeros(1,size(SNR_db,2));
BER_filter_2=zeros(1,size(SNR_db,2));
BER_theoretical=zeros(1,size(SNR_db,2));
No=zeros(1,size(SNR_db,2));
number_of_bits_received_wrong_from_matched_filter=zeros(1,size(SNR_db,2));
number_of_bits_received_wrong_from_filter_2=zeros(1,size(SNR_db,2));
for i=1:size(SNR_db,2)
    SNR_linear(1,i)=10^(SNR_db(1,i)/10);
    noise=randn(1,size_of_transmitted_signal); % AWGN with zero mean , variance =1
    No(1,i)=Eb/SNR_linear(1,i);
    BER_theoretical(1,i)=0.5*erfc( sqrt( Eb/No(1,i) ) );
    noise=noise*sqrt( No(1,i)/2 );
    transmitted_signal_with_noise=Transmitted_signal_for_10000_bits+noise;
    matched_filter_output_for_10000_bits=conv(transmitted_signal_with_noise,matched_filter);
    filter_2_output_for_10000_bits=conv(transmitted_signal_with_noise,filter_2);
    sampler=repmat([0,0,0,0,1],1,floor(size(matched_filter_output_for_10000_bits,2)/5)); % used to sample receiver signal every 5 samples
    sampler_size=size(sampler,2); % old size of sampler
    difference=size(matched_filter_output_for_10000_bits,2)-sampler_size;
    for k=1:difference
        sampler(1,sampler_size+k)=0;  % add zeros at the end of the sampler to equate between the sizes of the sampler and the received signal
    end
    received_signal_from_matched_filter_after_sampling_with_zeros=matched_filter_output_for_10000_bits.*sampler;
    received_signal_from_filter_2_after_sampling_with_zeros=filter_2_output_for_10000_bits.*sampler;
    received_signal_from_matched_filter_after_sampling=zeros(1,floor(size(received_signal_from_matched_filter_after_sampling_with_zeros,2)/5));
    received_signal_from_filter_2_after_sampling=zeros(1,floor(size(received_signal_from_filter_2_after_sampling_with_zeros,2)/5));
    for x=1:floor(size(received_signal_from_matched_filter_after_sampling_with_zeros,2)/5)  % remove zeros in this loop
        received_signal_from_matched_filter_after_sampling(1,x)=received_signal_from_matched_filter_after_sampling_with_zeros(1,5*x);
        received_signal_from_filter_2_after_sampling(1,x)=received_signal_from_filter_2_after_sampling_with_zeros(1,5*x);
    end
    for j=1:number_of_bits
        if (mapped_array_for_10000_bits(1,j)==1) && (received_signal_from_matched_filter_after_sampling(1,j)<0)
            number_of_bits_received_wrong_from_matched_filter(1,i) = number_of_bits_received_wrong_from_matched_filter(1,i) +1;
        elseif (mapped_array_for_10000_bits(1,j)== -1) && (received_signal_from_matched_filter_after_sampling(1,j)>0)
            number_of_bits_received_wrong_from_matched_filter(1,i) = number_of_bits_received_wrong_from_matched_filter(1,i) +1;
        end
        if (mapped_array_for_10000_bits(1,j)==1) && (received_signal_from_filter_2_after_sampling(1,j)<0)
            number_of_bits_received_wrong_from_filter_2(1,i) = number_of_bits_received_wrong_from_filter_2(1,i) +1;
        elseif (mapped_array_for_10000_bits(1,j)== -1) && (received_signal_from_filter_2_after_sampling(1,j)>0)
            number_of_bits_received_wrong_from_filter_2(1,i) = number_of_bits_received_wrong_from_filter_2(1,i) +1;
        end
    end
    BER_matched_filter(1,i)=number_of_bits_received_wrong_from_matched_filter(1,i)/number_of_bits;
    BER_filter_2(1,i)=number_of_bits_received_wrong_from_filter_2(1,i)/number_of_bits;
end
figure(3);
semilogy(SNR_db,BER_matched_filter,'b--');
title('BER ');
xlabel('Eb/No in db');
ylabel('BER');
grid on;
hold on;
semilogy(SNR_db,BER_filter_2,'r-');
hold on;
semilogy(SNR_db,BER_theoretical,'g:','linewidth',2);
legend('BER for matched filter','BER for rect filter','BER theoretical')
 
%%
%ISI and raised cosine part
number_of_bits=100;
array_of_100_random_bits = randi([0, 1], 1, number_of_bits);
 
mapped_array_for_100_bits = (2*array_of_100_random_bits)-1; %mapped 0 to -1 & mapped 1 to 1
 
sampled_array = upsample(mapped_array_for_100_bits,5);
Transmitted_signal = conv(sampled_array,pulse_shape);
 
% Define the parameters for the four cases
parameters = [
    0, 2; % Case a: R = 0, delay = 2
    0, 8; % Case b: R = 0, delay = 8
    1, 2; % Case c: R = 1, delay = 2
    1, 8  % Case d: R = 1, delay = 8
];
 
figure("name" , 'rcosine filter');
 
% Plot eye diagram for each case
for i = 1:size(parameters, 1)
    R = parameters(i, 1);
    delay = parameters(i, 2);
    
    % Generate square root raised cosine filter coefficients
    rcosine_filter = rcosine(1, 5, 'sqrt', R, delay);
    
    %plotting the rcosine filter
    nexttile; 
    plot(rcosine_filter);
    title("rcosine filter R : " + R + " Delay : " + delay);
    xlabel("Time of samples");
    ylabel("Amplitude");
    
    % Apply transmit filter
    Tx_rcosine_data = filter(rcosine_filter, 1, sampled_array);
    
    % Apply receive filter
    Rx_rcosine_data = filter(rcosine_filter, 1, Tx_rcosine_data);
    
    %sampling the recieved data
    train_of_pulses = repmat([0 0 0 0 1], 1, floor(size(Rx_rcosine_data, 2)/5));
    RX_sampled_data = Rx_rcosine_data .* train_of_pulses;
    
    
    % Plot eye diagram
    eyediagram_fig = eyediagram([Tx_rcosine_data ; Rx_rcosine_data]' , 10);
    set(eyediagram_fig,'Name',"eyediagram for R : " + R + " Delay : " + delay);
    f = figure;
    
end
close(f);
