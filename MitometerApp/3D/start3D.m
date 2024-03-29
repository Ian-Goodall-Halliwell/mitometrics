function [micronPerPixel, secondsPerFrame, zPlane, zDistance] = start3D(IJon)

if ~IJon
    prompt3D = {'What is your pixel size, in microns per pixel? ','What is your time between frames, in seconds? ', 'How many z-planes in the stack? ', 'What is the axial distance, in microns? '};
    dlgtitle3D = 'Start 3D';
    dims3D = [1 35];
    definput3D = {'0.18','10', '4', '0.3'};
    answer3D = inputdlg(prompt3D,dlgtitle3D,dims3D,definput3D);
    disp(dims3D)
    micronPerPixel = str2double(answer3D{1});
    secondsPerFrame = str2double(answer3D{2});
    zPlane = str2double(answer3D{3});
    zDistance = str2double(answer3D{4});
else
    prompt3D = {'What is your pixel size, in microns per pixel? ','What is your time between frames, in seconds? ', 'What is the axial distance, in microns? '};
    dlgtitle3D = 'Start 3D';
    dims3D = [1 35];
    definput3D = {'0.18','2.4','0.3'};
    answer3D = inputdlg(prompt3D,dlgtitle3D,dims3D,definput3D);
    micronPerPixel = str2double(answer3D{1});
    secondsPerFrame = str2double(answer3D{2});
    zDistance = str2double(answer3D{3});
    zPlane = nan;
end

end