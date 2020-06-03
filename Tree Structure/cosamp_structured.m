function Sest = cosamp_structured(Phi,u,K,tol,maxiterations,c,delta)

% Cosamp algorithm
%   Input
%       K : sparsity of Sest
%       Phi : measurement matrix
%       u: measured vector
%       tol : tolerance for approximation between successive solutions. 
%   Output
%       Sest: Solution found by the algorithm
%       c: constant bound sparsity
%       delta: constant control iteration number
%
% Algorithm as described in "CoSaMP: Iterative signal recovery from 
% incomplete and inaccurate samples" by Deanna Needell and Joel Tropp.
% 


% This implementation was written by David Mary, 
% but modified 20110707 by Bob L. Sturm to make it much clearer,
% and corrected multiple times again and again.
% To begin with, see: http://media.aau.dk/null_space_pursuits/2011/07/ ...
% algorithm-power-hour-compressive-sampling-matching-pursuit-cosamp.html
%
% This script/program is released under the Commons Creative Licence
% with Attribution Non-commercial Share Alike (by-nc-sa)
% http://creativecommons.org/licenses/by-nc-sa/3.0/
% Short Disclaimer: this script is for educational purpose only.
% Longer Disclaimer see  http://igorcarron.googlepages.com/disclaimer

% Embed Tree Structure by Yuchen Lou May 2020

% Initialization
Sest = zeros(size(Phi,2),1);
v = u;
t = 1; 
numericalprecision = 1e-14;
T = [];

while (t <= maxiterations) && (norm(v)/norm(u) > tol)
  y = abs(Phi'*v); % Why abs here?
 
  Omega = TreeApprox(y,2*K,c,2,delta);
  T = union(Omega,T);
  %b = pinv(Phi(:,T))*u;
  D = length(y);
  b = zeros(D,1);
  [b1,flag] = lsqr(Phi(:,T),u);
  b(T) = b1;
  Kgoodindices = TreeApprox(abs(b),K,c,2,delta);
  
  T = Kgoodindices;
  Sest = zeros(size(Phi,2),1);
  b = b(Kgoodindices);
  Sest(T) = b;
  
  v = u - Phi(:,T)*b;
  t = t+1;
  if t == maxiterations
      err = norm(v)/norm(u);
      disp(['Max iterations reached! Error is ', num2str(err)])
  end 
  if (norm(v)/norm(u) < tol)
      disp(['tolerance reached. Number of iterations is ',num2str(t)])
  end
end