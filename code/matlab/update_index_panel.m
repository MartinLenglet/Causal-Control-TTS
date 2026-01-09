function [m, n] = update_index_panel(m, n, y_nbr_graphs)
    if n+1 == y_nbr_graphs + 1
        m = m + 1;
    end
    n = mod(n+1, y_nbr_graphs+1);
    if n == 0
        n = 1;
    end
end