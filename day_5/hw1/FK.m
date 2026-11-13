%% ===============================================================
%  6-DOF Manipulator FK  (Classical / Standard DH)
%  A_i = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
%% ===============================================================
clear; clc; format short g
DEG2RAD = pi/180;

%% ---- DH table  [ d(mm) | theta_offset(deg) | a(mm) | alpha(deg) ] ----
DH_deg = [ 171.000     0.000     0.000    90.000 ;   % 1  yaw
    0.000     3.525   436.838     0.000 ;   % 2  pitch
    0.000    86.475    27.172   -90.000 ;   % 3  pitch
    380.727     0.000     0.000    90.000 ;   % 4  roll
    0.000     0.000     0.000   -90.000 ;   % 5  pitch
    0.000     0.000     0.000     0.000 ];  % 6  roll

P6    = [0; 0; 212.35];           % tool point in frame 6 [mm]
q_deg = [0, 0, 0, 0, 0, 0];       % all joints at zero
n     = size(DH_deg,1);

%% ---- UNIT CONVERSION : degree -> radian -----------------------
%  angle columns are 2(theta) and 4(alpha); 1(d) and 3(a) are lengths
DH_rad          = DH_deg;
DH_rad(:,[2 4]) = DH_deg(:,[2 4]) * DEG2RAD;
q_rad           = q_deg * DEG2RAD;

fprintf('=== deg -> rad ===\n');
for i = 1:n
    fprintf(' i=%d  th: %8.3f deg = %9.6f rad   al: %8.3f deg = %9.6f rad\n', ...
        i, DH_deg(i,2), DH_rad(i,2), DH_deg(i,4), DH_rad(i,4));
end

%% ---- forward kinematics --------------------------------------
A  = zeros(4,4,n);
T0 = zeros(4,4,n);
T  = eye(4);

for i = 1:n
    A(:,:,i)  = dhClassic( q_rad(i)+DH_rad(i,2), DH_rad(i,1), ...
        DH_rad(i,3),          DH_rad(i,4) );
    T         = T * A(:,:,i);
    T0(:,:,i) = T;
end

%% ---- print every transform -----------------------------------
for i = 1:n
    fprintf('\n---------------- i = %d ----------------\n', i);
    fprintf('A%d =\n', i);       disp( round(A(:,:,i), 4) );
    fprintf('T0_%d =\n', i);     disp( round(T0(:,:,i), 4) );
end

T06 = T;
P0  = T06 * [P6; 1];

fprintf('\n================ RESULT ================\n');
fprintf('T06 =\n');  disp( round(T06,4) );
fprintf('Frame6 origin [mm]: %9.3f %9.3f %9.3f\n', T06(1:3,4));
fprintf('Tool point    [mm]: %9.3f %9.3f %9.3f\n', P0(1:3));

%% ================= local function =================
function Ai = dhClassic(th, d, a, al)
ct = cos(th);  st = sin(th);
ca = cos(al);  sa = sin(al);
Ai = [ ct, -st*ca,  st*sa, a*ct ;
    st,  ct*ca, -ct*sa, a*st ;
    0,     sa,     ca,    d ;
    0,      0,      0,    1 ];
end