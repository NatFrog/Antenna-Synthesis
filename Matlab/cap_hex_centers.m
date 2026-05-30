function [theta0_list, phi0_list, seeds_xyz] = cap_hex_centers(N, thetaMaxDeg, nIter, nSamples)
% CAP_HEX_CENTERS  ~Equal-area "hex-like" centers on spherical cap 0<=theta<=thetaMaxDeg.
% theta measured from +z axis. thetaMaxDeg=70 means cap z in [cos70, 1].
% HEMISPHERE_HEX_CENTERS  ~Equal-area partition centers on upper hemisphere.
%   Returns N directions on z>=0 hemisphere. The Voronoi cells around these
%   directions are mostly hexagon-like after Lloyd relaxation.
%
% Inputs:
%   N        : number of centers (e.g., 100)
%   nIter    : Lloyd iterations (e.g., 10-20)
%   nSamples : Monte-Carlo points per iteration (e.g., 20000-80000)
%
% Outputs:
%   theta0_list (Nx1): degrees, [0..90] ideally
%   phi0_list   (Nx1): degrees, [-180..180)
%   seeds_xyz   (Nx3): unit vectors (x,y,z)
if nargin < 3, nIter = 12; end
if nargin < 4, nSamples = 40000; end

zmin = cosd(thetaMaxDeg);     % cap boundary in z
goldenAngle = pi*(3 - sqrt(5));
k = (0:N-1).';

% Equal-area on cap => z uniform in [zmin, 1]
z = zmin + (k + 0.5)/N * (1 - zmin);
r = sqrt(max(0, 1 - z.^2));
az = mod(k * goldenAngle, 2*pi);
seeds = [r.*cos(az), r.*sin(az), z];

% Lloyd relaxation on the cap
for it = 1:nIter
    % Sample uniformly on cap: z uniform in [zmin,1], phi uniform
    u  = zmin + (1 - zmin)*rand(nSamples,1);
    ph = 2*pi*rand(nSamples,1);
    rr = sqrt(max(0, 1 - u.^2));
    pts = [rr.*cos(ph), rr.*sin(ph), u];

    scores = pts * seeds.';          % dot products
    [~, idx] = max(scores, [], 2);

    sx = accumarray(idx, pts(:,1), [N 1], @sum, 0);
    sy = accumarray(idx, pts(:,2), [N 1], @sum, 0);
    sz = accumarray(idx, pts(:,3), [N 1], @sum, 0);
    cnt = accumarray(idx, 1,        [N 1], @sum, 0);

    empty = (cnt == 0);
    if any(empty)
        u2  = zmin + (1 - zmin)*rand(sum(empty),1);
        ph2 = 2*pi*rand(sum(empty),1);
        rr2 = sqrt(max(0, 1 - u2.^2));
        seeds(empty,:) = [rr2.*cos(ph2), rr2.*sin(ph2), u2];
        cnt(empty) = 1;
        sx(empty) = seeds(empty,1);
        sy(empty) = seeds(empty,2);
        sz(empty) = seeds(empty,3);
    end

    seeds = [sx./cnt, sy./cnt, sz./cnt];

    % Project to cap (enforce z>=zmin) and renormalize
    seeds(:,3) = max(seeds(:,3), zmin);
    seeds = seeds ./ vecnorm(seeds,2,2);
end

seeds_xyz = seeds;
theta0_list = rad2deg(acos(seeds(:,3)));
phi0_list   = rad2deg(atan2(seeds(:,2), seeds(:,1)));
phi0_list   = mod(phi0_list + 180, 360) - 180;
end
