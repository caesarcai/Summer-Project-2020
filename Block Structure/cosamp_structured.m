function Sest = cosamp_structured(Phi,u,K,tol,maxiterations,J,N)

% Cosamp algorithm
%   Input
%       K : sparsity of Sest
%       Phi : measurement matrix
%       u: measured vector
%       tol : tolerance for approximation between successive solutions. 
%   Output
%       Sest: Solution found by the algorithm
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

% Initialization
Sest = zeros(size(Phi,2),1);
v = u;
t = 1; 
numericalprecision = 1e-14;
T = [];
K = K/N;
D = J*N;

while (t <= maxiterations) && (norm(v)/norm(u) > tol)
  % CoSaMP using 1 norm? Why abs?
  
  
  y = abs(Phi'*v);
  Y = [];
  for i = 1:J
     a = y((1+N*(i-1)):(N+N*(i-1)));
     Y = [Y norm(a,2)];
  end
  [vals,z] = sort(Y,'descend');
  Omega = find(Y >= vals(2*K) & Y > numericalprecision);
  T = union(Omega,T);
  T1  = [];
  for i = 1:length(T)
     T1 = [T1 (1+(T(i)-1)*N : N+(T(i)-1)*N)]; 
  end
  
  %b = pinv(Phi(:,T))*u;
  [b,flag] = lsqr(Phi(:,T1),u);
  
  B = [];
  for i = 1:(length(b)/N)
     c = b((1+N*(i-1)):(N+N*(i-1)));
     B = [B norm(c,2)];
  end
  [vals,z] = sort(B,'descend');
  Kgoodindices = find(B >= vals(K) & B > numericalprecision);
  T = T(Kgoodindices);
  Sest = zeros(size(Phi,2),1);
  
  index = [];
  for i = 1:length(T)
      index = [index (1+N*(Kgoodindices(i)-1)):(N+N*(Kgoodindices(i)-1))];
  end
  
  b = b(index);
  T1  = [];
  for i = 1:length(T)
     T1 = [T1 (1+(T(i)-1)*N : N+(T(i)-1)*N)]; 
  end
  v = u - Phi(:,T1)*b;
  Sest(T1) = b;
  
  t = t+1;
  if t == maxiterations
      err = norm(v)/norm(u);
      disp(['Max iterations reached! Error is ', num2str(err)])
  end 
  if (norm(v)/norm(u) < tol)
      disp(['tolerance reached. Number of iterations is ',num2str(t)])
  end
end