


%% 2D arbitrary potential wave packet simulator

clear

% read potential shape from image
image = imread('potential.png');
image = double(image(:,:,1));

% SIMULATION PARAMETERS
Nx = size(image,2);
Ny = size(image,1);
Nt = 2000; % number temporal points 

batchSize = 1000; % batch size for GPU, adjust based on available memory
Nbatches = ceil(Nt/batchSize);
NtBatch = round(Nt/Nbatches);

plotinterval = 2; %  step interval for animation
Lx = 200/sqrt(2); % physical length of the grid
Ly = Lx * (Ny/Nx);

Lb = 20; %width of the box for square well


dx = Lx/(Nx-1); % resulting spatial resolution
dy = Ly/(Ny-1);

tmax = 20; % temporal length of simulation
dt = tmax/(Nt-1); % resulting timestep

% WAVE PACKET / POTENTIAL PARAMETERS
x0 = 0.2*Lx; % position of initial wave packet
y0 = Ly/2;

sigmax = 3.0; % width of initial wave packet
sigmay = 3.0;

Vc = 2/Lx; % potential strength

k0x = -0.5; % initial momentum of the wave packet
k0y = 0;

Voutside = 1000-5i; %potential barrier height
Vdecoration = 1000-5i; %potential barrier height


% BUILD WAVE PACKET / POTENTIAL
cx = 1/(2*sigmax^2); % constant for Gaussian wave packet
cy = 1/(2*sigmay^2);


imtime = false; %imaginary time converges to ground state
loadFunMode = false; %load wave function from file
makeVideo = true; %write output to video file
videoName = 'out.avi'; %output video file name

showAnimation = false; %show animation on screen

if imtime
    tmax = 0-1i*tmax;
    dt = 0-1i*dt;
end


%% Define potential

%[V, V2D] = squareWell(Vh,L,Lb,dx,N);
%[V, V2D] =parabolicWell(Vh,L,dx,N);

% [V, V2D] = doubleslitPotential(Vh,Lx,Ly,dx,Nx,Ny);


V2D = zeros(size(image));
V2D(image<250) = Vdecoration;
V2D(image<100) = Voutside;

xsmall = linspace(1,Nx,Nx/5);
ysmall = linspace(1,Ny,Ny/5);

V2D_small = interp2(1:Nx,1:Ny,real(V2D),xsmall,ysmall');

% smooth potential
% V2D = imgaussfilt(real(V2D),10) + 1i*imgaussfilt(imag(V2D),10);

% plot potential
figure(6);
clf;
surf((abs(V2D)),'EdgeColor','none')

% plot potential eigenstates
%plotEigenstates(V2D_small,length(xsmall),length(ysmall));

%% initial gaussian wave packet
for ix=1:Nx
    phix_k(ix) = exp(1i*k0x*(dx*(ix-1)-x0));
    phix(ix) = sqrt(Vc)*((cx/pi)^(1/4)*exp(1i*k0x*(dx*(ix-1)-x0)) * exp(-(cx/2)*(dx*(ix-1)-x0)^2));


    V(ix) = 1e1*(exp(-(dx*(ix-1)-10))+ exp((dx*(ix-1)-(Lx-10))));
end

for iy = 1:Ny

    phiy_k(iy) = exp(1i*k0y*(dy*(iy-1)-y0));
    phiy(iy) = sqrt(Vc)*((cy/pi)^(1/4)*exp(1i*k0y*(dy*(iy-1)-y0)) * exp(-(cy/2)*(dy*(iy-1)-y0)^2));

end

%phiy = ones(size(phix));
%phiy(logical(V2D(:,N/2))) = 0;
phi2D = phix.*phiy';


if loadFunMode
    funMode = load('funMode.mat');
    phi2D = funMode.funMode;
end

phi2D = phi2D.*phix_k.*phiy_k';

% plot initial state
% fig = figure(1);
% clf
% subplot(1,2,1)
% s1 = surf(abs(phi2D(:,:,1)),'EdgeColor','none');
% subplot(1,2,2)
% s1 = surf(abs(fftshift(fft2(phi2D(:,:,1)))),'EdgeColor','none');

%% Time evolution Computation

kL = sqrt(2)*Lx;

k = zeros(1,Nx);
for n=1:Nx
    if n<=0.5*Nx
        k(n)=(n-1)*(2*pi/kL);
    else
        k(n)=(n-1-Nx)*(2*pi/kL);
    end
end



% Time evolution operators
r = linspace(-kL/2,kL/2,Nx);

expidtV = exp(-1i*dt*V);
expikdt = exp(-1i*(k).^(2)*dt/2);

x = linspace(-Lx/2,Lx/2,Nx);
y = linspace(-Ly/2,Ly/2,Ny)';

r2d = sqrt(x.^2+y.^2);

expik2dt = interp1(r,expikdt,r2d);
expidt2dV = exp(-1i*dt*V2D);


% move data to GPU
phiGPU = gpuArray(phi2D);
expik2dtGPU  = gpuArray(expik2dt);
expidt2dVGPU = gpuArray(expidt2dV);


phiGPU = fftshift(fft2(phiGPU));

if makeVideo

    v = VideoWriter(videoName);
    v.FrameRate = 30;
    open(v);
end

% Time evolution loop using the split operator method

for batchIdx = 1:Nbatches

i=1;
j=1; % loop indices

PhiOut = complex(zeros(Ny,Nx,NtBatch/plotinterval));
if ~imtime
    tvec = 0:dt:round(tmax/Nbatches);
else
    tvec = 0+ (0:abs(dt):abs(round(tmax/Nbatches)))*1i;
end



for tstep=tvec
    i=i+1;

    %momentum space
    phiGPU = phiGPU.*expik2dtGPU;
    phiGPU = ifft2(ifftshift(phiGPU));

    % Position space
    phiGPU = phiGPU.*expidt2dVGPU;


    % Store wavepacket in plotintervals
    if ~mod(i,plotinterval)
        time2(j) = tstep;
        PhiOut(:,:,j) = gather(phiGPU);
        j=j+1;
        tstep
    end

    %renormalization for imaginary time
    if imtime

        renorm_factor = sum(abs(phiGPU).^2,'all')*dx;
        phiGPU = phiGPU./sqrt(renorm_factor);

    end

    phiGPU = fftshift(fft2(phiGPU));
    %momentum space
    phiGPU = phiGPU.*expik2dtGPU;

end


if imtime

    funMode = PhiOut(:,:,end);
    save('funMode','funMode');

end

%% render video
fig = figure('Visible','off');
clf

s1 = surf(abs(PhiOut(:,:,1)),'EdgeColor','none');
ax = gca;

ax.ZLimMode = 'manual';
zmax = 2*max(max(max(abs(PhiOut).^2)));
zmax = max(zmax,0.01);
ax.ZLim = [0 zmax];


c1 = [0 0 0];
c2 = [0 1 0];
c3 = [1 1 0];
c4 = [1 0 0];

xpos = [1 2 3];
cpos = [1 96 192 256];

[XQ YQ] = meshgrid([1 2 3],1:256);

c = interp2(xpos,cpos,[c1;c2;c3;c4],XQ,YQ);

%c = circshift(colormap(hot(256)),2,2);
interpscale = linspace(0,1,length(c)).^3;
interpR = griddedInterpolant(interpscale,c(:,1),'linear','nearest');
interpG = griddedInterpolant(interpscale,c(:,2),'linear','nearest');
interpB = griddedInterpolant(interpscale,c(:,3),'linear','nearest');

view(2)

if showAnimation
    fig.Visible = 'on';
end


phiScale = 2.0e-7;

% V2Dframe = V2D/max(max(V2D));
V2Dframe = zeros(size(V2D));

for tstep=1:NtBatch/plotinterval
    %
    if makeVideo
        fprintf('writing frame %d / %d \n',tstep,NtBatch/plotinterval);
        phiFrame = abs(PhiOut(:,:,tstep)).^2;

        %           phiScale = max(phiFrame,[],'all');

        phiFrame = phiFrame./phiScale;

        R = interpR(phiFrame);
        G = interpG(phiFrame);
        B = interpB(phiFrame);


        %           R = min(R+V2Dframe,1);
        %           G = min(G+V2Dframe,1);
        %           B = min(B+V2Dframe,1);


        RGBframe(:,:,1) = R;
        RGBframe(:,:,2) = G;
        RGBframe(:,:,3) = B;

        writeVideo(v,RGBframe);

    end

    if showAnimation

        s1.ZData = abs(PhiOut(:,:,tstep)).^2;
        drawnow


    end
end
clear("PhiOut");
end


if makeVideo

    close(v)

end

%%
function [V, V2D] = squareWell(Vh,L,Lb,dx,N)

    V = zeros(1,N);
    
    Vlim = round((L-Lb)/(2*dx));
    
    
    V(1:Vlim) = Vh;
    V(end-Vlim:end) = Vh;
    
    % figure(3)
    % clf
    V = V-min(V);
    
    V2D = zeros(N);%-0.001i;
    V2D(1:Vlim,:) = Vh;
    V2D(end-Vlim:end,:) = Vh;
    V2D(:,1:Vlim) = Vh;
    V2D(:,end-Vlim:end) = Vh;

end

function [V, V2D] = parabolicWell(Vh,L,dl,Nx,Ny)
    
    V = zeros(Nx,Ny);
    
    x = [-L/2 0 L/2];
    y = [Vh 0 Vh];
    
    p = polyfit(x,y,2);
    
    parabola = @(x) p(1).*x.^2+p(2).*x+p(3);
    
    V = parabola(0:dl:L);
    
    
    [X, Y] = meshgrid(-L/2:dl:L/2,-L/2:dl:L/2);
    r = sqrt(X.^2+Y.^2);
    
    V2D = parabola(r);


end

function [V, V2D] = doubleslitPotential(Vh,Lx,Ly,dl,Nx,Ny)

    V = zeros(Nx,Ny);
    
    x = [-Lx/2 0 Lx/2];
    y = [Vh 0 Vh];
    
    [X, Y] = meshgrid(-Lx/2:dl:Lx/2,-Ly/2:dl:Ly/2);
    
    V2D = zeros(size(X));
    
    barrierPos = [0.08*Lx 0.1*Lx];
    
    barrierIdx = X(1,:)>barrierPos(1) & X(1,:)<barrierPos(2);
    V2D(:,barrierIdx) = Vh;
    
    slitCenters = [-0.05*Ly 0.05*Ly];
    slitWidth = 0.04*Ly;
    
    for ii = 1:length(slitCenters)
        slitPos(ii,:) = [slitCenters(ii)-slitWidth/2 slitCenters(ii)+slitWidth/2];
    
    
        slitIdx = Y(:,1)>slitPos(ii,1) &  Y(:,1)<slitPos(ii,2);
    
        V2D(slitIdx,:) = 0;
    
    end
    
    
    % box boundary
    Vlim = 25;
    
    V2D(1:Vlim,:) = 1*Vh;
    V2D(end-Vlim:end,:) = 1*Vh;
    V2D(:,1:Vlim) = 1*Vh;
    V2D(:,end-Vlim:end) = 1*Vh;




end
function plotEigenstates(V2D,Nx,Ny)
    Hsize = numel(V2D);
    
    %main diagonal
    H = sparse(Hsize,Hsize);
    I = 1:Hsize;
    v = 2*ones(size(I));
    H = H+sparse(I,I,v);
    
    H = H+sparse(I,I,-1*real(V2D(:))+real(max(max(V2D))));
    
    %off diagonals
    
    J = I+1;
    J(end) = 1;
    v = -1*ones(size(I));
    H = H+sparse(I,J,v);
    H = H+sparse(J,I,v);
    
    % 2d diagonals
    I2 = I+Ny;
    I2(I2>Hsize) = I2(I2>Hsize)-Hsize;
    v = ones(size(I));
    
    H = H+sparse(I,I2,v);
    H = H+sparse(I2,I,v);
    
    %%
    [V,D] = eigs(H);
    
    figure(7);
    clf
    for iii = 1:6
        subplot(2,3,iii);
        plotmag = abs(reshape(V(:,iii),Ny,Nx));
        plotmag(V2D>1) = 1.2*max(max(plotmag));
        surf(plotmag,'edgecolor','none');
        view(2);
        colormap(hot)
    end
end


