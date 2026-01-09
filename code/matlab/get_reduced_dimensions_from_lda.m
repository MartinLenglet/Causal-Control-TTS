function [lambda, W, Z] = get_reduced_dimensions_from_lda(Mdl, X)
    [W, LAMBDA] = eig(Mdl.BetweenSigma, Mdl.Sigma); %Must be in the right order! 
    lambda = diag(LAMBDA);
    [lambda, SortOrder] = sort(lambda, 'descend');
    W = W(:, SortOrder);
    Y = X*W;
    Z = Y\X;
end