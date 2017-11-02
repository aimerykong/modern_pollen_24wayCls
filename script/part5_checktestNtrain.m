clear
close all
clc;

imgFolderName = 'DBtest_24way';

%% read training image list
trfileName = 'DBtrain24Way_thresh4.txt';
fn = fopen(trfileName, 'r');

imgListDict = containers.Map;

tline = fgets(fn);
while ischar(tline)
    C = strsplit(tline, ' ');
    [imgPath, imgName, imgExt] = fileparts(C{1});
    imgLabel = str2double(C{2});
    
    aa = strsplit(imgName, '_');
    
    if ~isKey(imgListDict, aa{1})
        imgListDict(aa{1}) = imgLabel;
    end
    tline = fgets(fn);
end
fclose(fn);

%% check testing images
count = 0;
className = dir(imgFolderName);
className = className(3:end);
testNameList = {};
for c = 1:length(className)
    imList = dir( fullfile(imgFolderName,className(c).name, '*.jpg') );
    for i = 1:length(imList)        
        imgName = imList(i).name;
        [imgPath, imgName, imgExt] = fileparts(imgName);
        aa = strsplit(imgName, '_');
        testNameList{end+1} = aa{1};
        if isKey(imgListDict, aa{1})
            count = count + 1;
            fprintf('%s\n', imgName);
        end
    end
end

%%