function fError=CheckNumber(numToCheck)
fError = 0;
if(any(isnan(numToCheck)))
    fError = 1;
    error('Nan %g', numToCheck);    
end
if(any(isinf(numToCheck)))
    fError = 1;
    error('Inf %g', numToCheck);
end
