function R=renyientropy(W,alpha)
% R=renyientropy(W,alpha)
% calculates pixelwise R�nyi entropy from the pseudo-Wigner distribution W 
% produced by "localwigner"
% Inputs:
% W: pseudo-Wigner distribution of a given image by function "localwigner"
% alpha: shape parameter (use a positive integer: 1, 2, 3, ...).
% By omission alpha=3 is used. When alpha=1 the outcome will be 
% the Shannon entropy.
% Outputs:
% R: pixel-wise R�nyi entropy (or Shannon if alpha=1) of the image
% in the direction given by W
% By Salvador Gabarda 2011
% salvador@optica.csic.es
 
 
% normalize pseudo-Wigner distribution
[ro co N]=size(W);    
W2=reshape(W,ro*co,N);
P=W2.*conj(W2);
S=sum(P,2);
SS=repmat(S,1,N);
P=P./(SS+eps);

if nargin==1
    alpha=3;
end

if alpha==1
    % R�nyi entropy = Shannon entropy
    Pp=P.*log2(P+eps);
    Q=-sum(Pp,2);
else
    % R�nyi entropy
    P=P.^alpha;
    Q=(1/(1-alpha))*log2(sum(P,2)+eps);
end

% round-off error correction
I=find(Q<0); 
Q(I)=0;
II=find(Q>log2(N));
Q(II)=0;
U=reshape(Q,ro,co);
R=U./log2(N);
 
