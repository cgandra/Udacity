# CFAR

Pls note that the code has been restructured to make it easier to loop over multiple initial pos/vel values and make the code layout more modular
Secondly I'd like to note that these projects seem to be geared more towards RCGs rather than imaging/cv/deep learning professionals

1. Implementation steps for the 2D CFAR process
   * Set the number of Training and Guard Cells in both range and doppler dimensions
   * For the desired false alarm probability (Pfa) compute the offset as below and convert to db
    ```
     offset = num_cells*(Pfa^(-1/num_cells)-1)
    ```
   * There are 2 versions of code for 2D CFAR, cfar_2d and cfar_2d_vec. The latter being the vectorized version
   * Convert the RDM signal to linear RDMp using db2pow
   * Compute the sum of grid (training+guard) window centered @ each cell in RDMp with grid window of [2*(Tr+Gr)+1, 2*(Td+Gd)+1]
   * Compute the sum of guard window centered @ each cell in RDMp with guard window of [2*Gr+1, 2*Gd+1]
   * Compute sum of training cells = sum of grid window-sum of guard window
   * Compute mean of training cells, convert to db and add threshold in db
   * Set cfar out signal to 1 when the CUT signal value > threshold

2. Selection of Training, Guard cells and offset
   * In vectorized version both the grid & guard windows are selected in the func 
     blockproc using BorderSize & TrimBorder property
    ```
     grid_sum = blockproc(RDMp, [1 1], @(x) sum(x.data(:)), 'BorderSize', [Gr+Tr Gd+Td], 'TrimBorder', false, 'PadPartialBlocks', true);
     guard_sum = blockproc(RDMp, [1 1], @(x) sum(x.data(:)), 'BorderSize', [Gr Gd], 'TrimBorder', false, 'PadPartialBlocks', true);
    ```
   * In non vectorized version both the grid & guard windows are selected via looping
     over the valid ranges of grid/guard cells to gather grid and gaurd cell sums to 
     compute average of training cells
    ```
    for d = d_min:d_max 
       for r = r_min:r_max
          % Use RDM[x,y] as the matrix from the output of 2D FFT for implementing
          % CFAR
          full_win  = RDMp(r-(Gr+Tr):r+Gr+Tr, d-(Gd+Td):d+Gd+Td);
          guard_win = RDMp(r-Gr:r+Gr, d-Gd:d+Gd);
          noisePowEst = sum(full_win(:))-sum(guard_win(:));
          win_sz = length(full_win(:))-length(guard_win(:));
          noisePowEst = noisePowEst/win_sz;
    ```
   * For the desired false alarm probability (Pfa) offset was computed as below and convert to db
    ```
     offset = num_cells*(Pfa^(-1/num_cells)-1)
    ```
3. Steps taken to suppress the non-thresholded cells at the edges
   * In vectorized version only CUT signal to threshold comparison is done only for valid range
    ```
   signal_cfar(r_min:r_max, d_min:d_max) = (RDM(r_min:r_max, d_min:d_max) >= threshold(r_min:r_max, d_min:d_max));
    ```

   * In non vectorized version again only valid CUT cells range is used for main looping 
    ```
    for d = d_min:d_max
       for r = r_min:r_max
          % Use RDM[x,y] as the matrix from the output of 2D FFT for implementing
          % CFAR
          full_win  = RDMp(r-(Gr+Tr):r+Gr+Tr, d-(Gd+Td):d+Gd+Td);
          guard_win = RDMp(r-Gr:r+Gr, d-Gd:d+Gd);
          noisePowEst = sum(full_win(:))-sum(guard_win(:));
          win_sz = length(full_win(:))-length(guard_win(:));
          noisePowEst = noisePowEst/win_sz;
    ```