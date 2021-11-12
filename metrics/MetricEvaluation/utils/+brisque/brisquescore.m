function qualityscore  = brisquescore(imdist,temp_path)

import brisque.*;

if(size(imdist,3)==3)
    imdist = uint8(imdist);
    imdist = rgb2gray(imdist);
end

imdist = double(imdist);

if(nargin<3)
feat = brisque_feature(imdist);
disp('feat computed')
end


%---------------------------------------------------------------------
%Quality Score Computation
%---------------------------------------------------------------------
temp_path=['.\MetricEvaluation\utils\temp\',temp_path];
mkdir(temp_path);
fid = fopen([temp_path,'/test_ind'],'w');

for jj = 1:size(feat,1)
    
fprintf(fid,'1 ');
for kk = 1:size(feat,2)
fprintf(fid,'%d:%f ',kk,feat(jj,kk));
end
fprintf(fid,'\n');
end

fclose(fid);
warning off all

system(['.\MetricEvaluation\utils\+brisque\svm-scale -r .\MetricEvaluation\utils\+brisque\allrange ',temp_path,'\test_ind >> ',temp_path,'\test_ind_scaled']);
system(['.\MetricEvaluation\utils\+brisque\svm-predict -b 1 ',temp_path,'\test_ind_scaled .\MetricEvaluation\utils\+brisque\allmodel ',temp_path,'\output >>',temp_path,'\dump']);

load([temp_path,'\output']);

qualityscore = output;
fclose all;
rmdir(temp_path,'s') ;

