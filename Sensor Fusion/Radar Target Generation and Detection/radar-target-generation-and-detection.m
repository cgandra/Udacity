clear all;
close all;
clc;

%% Radar Specifications 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency of operation = 77GHz
% Max Range = 200m
% Range Resolution = 1 m
% Max Velocity = 70 m/s
% Velocity Resolution = 3 m/s
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%speed of light = 3e8

disp_res = true;

%% User Defined Range and Velocity of target
% *%TODO* :
% define the target's initial position and velocity. Note : Velocity
% remains contant

% You will provide the initial range and velocity of the target. 
% Range cannot exceed the max value of 200m and velocity can be any value in the range of -70 to + 70 m/s.
run_all_pos_vel = false;
if run_all_pos_vel
    for init_pos=5:10:200
        for init_vel=-70:5:70
            res = ['Pos: ', num2str(init_pos),' Vel: ', num2str(init_vel)];
            disp(res)
            process(init_pos, init_vel, disp_res);
        end
    end
else
    init_pos = 200;
    init_vel = 70;
    res = ['Pos: ', num2str(init_pos),' Vel: ', num2str(init_vel)];
    disp(res);
    process(init_pos, init_vel, disp_res);
end



function process(init_pos, init_vel, disp_res)
c = 3e8; % m/s
max_range = 200; % m
max_vel = 70; % m/s
range_res = 1; % m
vel_res = 3; % m/s

%% FMCW Waveform Generation

% *%TODO* :
% Design the FMCW waveform by giving the specs of each of its parameters.
% Calculate the Bandwidth (B), Chirp Time (Tchirp) and Slope (slope) of the FMCW
% chirp using the requirements above.

Bsweep = c/(2*range_res); % Bandwidth (Hz)
Tchirp = (5.5*2*max_range)/c; % Chirp duration (s)
sweepSlope = Bsweep/Tchirp; % FMCW sweep slope (Hz/s)

%Operating carrier frequency of Radar 
fc= 77e9;             % carrier freq in Hz
                                                          
%The number of chirps in one sequence. Its ideal to have 2^ value for the ease of running the FFT
%for Doppler Estimation. 
Nd=128;               % #of doppler cells OR #of sent periods % number of chirps

%The number of samples on each chirp. 
Nr=1024;              % for length of time OR # of range cells

% Timestamp for running the displacement scenario for every sample on each
% chirp
t=linspace(0,Nd*Tchirp,Nr*Nd); % total time for samples


%% Signal generation and Moving Target simulation
% Running the radar scenario over the time. 
% For each time stamp update the Range of the Target for constant velocity.
r_t = init_pos + init_vel*t; %R(t) = R0 + vt, range of the target in terms of its velocity and initial range
td = (2*r_t/c); % delay time for received signal
tr = t-td;
% For each time sample we need update the transmitted and received signal
Tx = cos(2*pi*(fc*t + (sweepSlope*t.*t/2)));
Rx = cos(2*pi*(fc*tr + (sweepSlope*tr.*tr/2)));

% *%TODO* :
% Now by mixing the Transmit and Receive generate the beat signal
% This is done by element wise matrix multiplication of Transmit and
% Receiver Signal
Mix = Tx.*Rx;


%% RANGE MEASUREMENT

% *%TODO* :
% reshape the vector into Nr*Nd array. Nr and Nd here would also define the size of
% Range and Doppler FFT respectively.
Mix =reshape(Mix,[Nr,Nd]);

% *%TODO* :
% run the FFT on the beat signal along the range bins dimension (Nr) and
% normalize.
signal_fft = fft(Mix,Nr)/Nr;

% *%TODO* :
% Take the absolute value of FFT output
P2 = abs(signal_fft);

% *%TODO* :
% Output of FFT is double sided signal, but we are interested in only one side of the spectrum.
% Hence we throw out half of the samples.
P1 = P2(1:(Nr/2)+1, :);

%plotting the range
% *%TODO* :
% plot FFT output 
if disp_res
    figure ('Name','Range from First FFT')
    fs = (0:(Nr/2));
    subplot(2,1,1);
    plot(fs,P1)
    axis ([0 200 0 0.5]);
    subplot(2,1,2);
    plot(fs,P1(:, 1))
    axis ([0 200 0 0.5]);
end

[r,~] = find(P1>=max(P1));
res = ['Range FFT min_rng: ', num2str(min(r)-1),' max_rng: ', num2str(max(r)-1)];
disp(res);


%% RANGE DOPPLER RESPONSE
% The 2D FFT implementation is already provided here. This will run a 2DFFT
% on the mixed signal (beat signal) output and generate a range doppler
% map.You will implement CFAR on the generated RDM


% Range Doppler Map Generation.

% The output of the 2D FFT is an image that has reponse in the range and
% doppler FFT bins. So, it is important to convert the axis from bin sizes
% to range and doppler based on their Max values.

% 2D FFT using the FFT size for both dimensions.
sig_fft2 = fft2(Mix,Nr,Nd);

% Taking just one side of signal from Range dimension.
sig_fft2 = sig_fft2(1:Nr/2,1:Nd);
sig_fft2 = fftshift(sig_fft2);
RDM = abs(sig_fft2);
RDM = 10*log10(RDM) ;

%use the surf function to plot the output of 2DFFT and to show axis in both
%dimensions
doppler_axis = linspace(-100,100,Nd);
range_axis = linspace(-200,200,Nr/2)*((Nr/2)/400);
if disp_res
    figure,surf(doppler_axis,range_axis,RDM);
end

%% CFAR implementation

%Slide Window through the complete Range Doppler Map

% *%TODO* :
%Select the number of Training Cells in both the dimensions.
Tr = 10;
Td = 8;

% *%TODO* :
%Select the number of Guard Cells in both dimensions around the Cell under 
%test (CUT) for accurate estimation
Gr = 4;
Gd = 4;

% Total num of cells in grid used for thresholding
num_cells = (2*(Tr+Gr)+1)*(2*(Td+Gd)+1)-((2*Gr+1)*(2*Gd+1));
% desired false alarm probability
Pfa = 5e-7;
% *%TODO* :
% offset the threshold by SNR value in dB
offset = pow2db(num_cells*(Pfa^(-1/num_cells)-1));
%offset = 6;

% *%TODO* :
%design a loop such that it slides the CUT across range doppler map by
%giving margins at the edges for Training and Guard Cells.
%For every iteration sum the signal level within all the training
%cells. To sum convert the value from logarithmic to linear using db2pow
%function. Average the summed values for all of the training
%cells used. After averaging convert it back to logarithimic using pow2db.
%Further add the offset to it to determine the threshold. Next, compare the
%signal under CUT with this threshold. If the CUT level > threshold assign
%it a value of 1, else equate it to 0.

signal_cfar = cfar_2d_vec(Tr, Td, Gr, Gd, RDM, offset);
[r, c] = find(signal_cfar>0);

res = ['cfar2d min_rng: ', num2str(min(range_axis(r))),' max_rng: ', num2str(max(range_axis(r)))];
disp(res);
res = ['cfar2d min_vel: ', num2str(min(doppler_axis(c))),' max_vel: ', num2str(max(doppler_axis(c)))];
disp(res);
if disp_res
    figure,surf(doppler_axis,range_axis,signal_cfar);
    colorbar;
end
end

function [r_min,r_max,d_min,d_max] =  get_ranges(Nr, Nd, Tr, Td, Gr, Gd)
r_max = Nr-(Gr+Tr);
r_min = Tr+Gr+1;

d_max = Nd-(Gd+Td);
d_min = Td+Gd+1;
end

function signal_cfar = cfar_2d_vec(Tr, Td, Gr, Gd, RDM, offset)
RDMp = db2pow(RDM);
[Nr, Nd] = size(RDM);
[r_min,r_max,d_min,d_max] = get_ranges(Nr, Nd, Tr, Td, Gr, Gd);

%Vector to hold final signal after thresholding
signal_cfar = zeros(Nr, Nd);
grid_sum = blockproc(RDMp, [1 1], @(x) sum(x.data(:)), 'BorderSize', [Gr+Tr Gd+Td], 'TrimBorder', false, 'PadPartialBlocks', true);
guard_sum = blockproc(RDMp, [1 1], @(x) sum(x.data(:)), 'BorderSize', [Gr Gd], 'TrimBorder', false, 'PadPartialBlocks', true);
noisePowEst = grid_sum-guard_sum;
num_cells = (2*(Tr+Gr)+1)*(2*(Td+Gd)+1)-((2*Gr+1)*(2*Gd+1));
noisePowEst = noisePowEst/num_cells;
threshold = pow2db(noisePowEst)+offset;
signal_cfar(r_min:r_max, d_min:d_max) = (RDM(r_min:r_max, d_min:d_max) >= threshold(r_min:r_max, d_min:d_max));
end


function signal_cfar = cfar_2d(Tr, Td, Gr, Gd, RDM, offset)
RDMp = db2pow(RDM);
[Nr, Nd] = size(RDM);
[r_min,r_max,d_min,d_max] = get_ranges(Nr, Nd, Tr, Td, Gr, Gd);

%Vector to hold final signal after thresholding
signal_cfar = zeros(Nr, Nd);

for d = d_min:d_max 
    for r = r_min:r_max
        % Use RDM[x,y] as the matrix from the output of 2D FFT for implementing
        % CFAR
        full_win  = RDMp(r-(Gr+Tr):r+Gr+Tr, d-(Gd+Td):d+Gd+Td);
        guard_win = RDMp(r-Gr:r+Gr, d-Gd:d+Gd);
        noisePowEst = sum(full_win(:))-sum(guard_win(:));
        win_sz = length(full_win(:))-length(guard_win(:));
        noisePowEst = noisePowEst/win_sz;
        noisePowEst = pow2db(noisePowEst);
        threshold = (noisePowEst + offset);

        % 6. Measuring the signal within the CUT
        signal = RDM(r, d);

        % 8. Filter the signal above the threshold
        if signal >= threshold
            signal_cfar(r, d) = 1;
        end
    end
end
end
