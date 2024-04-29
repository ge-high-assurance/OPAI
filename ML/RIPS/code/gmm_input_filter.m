function [qual_data_filt] = gmm_input_filter(qual_data_x,trn_gmm)
%GMM_INPUT_FILTER Summary of this function goes here
%Apply GMM model to qualification data and return Yes/No vector for data samples
qual_data_filt = zeros(size(qual_data_x,1),1);
            for i=1:size(qual_data_x,1)
            [~,nl(i)] = posterior(trn_gmm,qual_data_x(i,:));
                if nl(i) <= 42.7408
                qual_data_filt(i)=1;
                end
            end
end

