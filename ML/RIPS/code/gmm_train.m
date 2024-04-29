function [gmm_model,qual_filt,nl_th] = gmm_train(trn_data_x, qual_data_x, gmm_trnflag, nummodes_mat)
%Train GMM if trnflag is set to 1 and obtain best model
    if gmm_trnflag==1
        nmodes = max(nummodes_mat) - min(nummodes_mat) + 1
        AIC = zeros(1,nmodes);
        GMModels = cell(1,nmodes);
        options = statset('MaxIter',1000);
        rng(26);
            for k = 1: nmodes
                k
                GMModels{k} = fitgmdist(trn_data_x,k, 'Options', options);
                AIC(k)= GMModels{k}.AIC;
            end
        figure()
        plot(1:nmodes,AIC)    
        [~,numComponents] = min(AIC);
        
        numComponents
        %best GMM model
%        gmdl = GMModels{numComponents};
         gmdl = fitgmdist(trn_data_x,numComponents, 'Options', options);
        
         if size(qual_data_x,2)==2
            figure()
            ezcontour(@(x1,x2)pdf(gmdl,[x1 x2]),get(gca,{'XLim','YLim'}));
            title('Training data region');
            grid on;
            hold on
            plot(qual_data_x(:,1), qual_data_x(:,2),'k.')
            plot(trn_data_x(:,1), trn_data_x(:,2),'cx')
         end

        %Estimate threshold given GMM model and training data
            for i=1:size(trn_data_x,1)
                [~,nl(i)] = posterior(gmdl,trn_data_x(i,:));
            end
        nl_th=max(nl)

        %Apply GMM model to qualification data and return Yes/No vector for data samples
        qual_filt = zeros(size(qual_data_x,1),1);
            for i=1:size(qual_data_x,1)
            [~,nl(i)] = posterior(gmdl,qual_data_x(i,:));
                if nl(i) <= nl_th
                qual_filt(i)=1;
                         if size(qual_data_x,2)==2
                            plot(qual_data_x(i,1), qual_data_x(i,2),'ro')
                         end
                end
            end
    end
end

