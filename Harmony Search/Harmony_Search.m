%%
%%-------------------------------------------------------------------------
% This is a simple demo version implementing the basic Harmony Search      %
% algorithm for clustering problem without fine-tuning the parameters.     %
% Though this demo works very well, it is expected that this illustration  %  
% is much less efficient than the work reported in paper:                  %
% (Citation details):                                                      % 
% J. Senthilnath, Sushant Kulkarni, Raghuram, D.R., Sudhindra, M.,         %
% Omkar, S.N., Das, V. and Mani, V (2016) "A novel harmony search      %
% based approach for clustering problems", Int. J. Swarm Intelligence,     %
%  Vol. 2, No. 1, pp.66 � 86, 2016.                                           %
%%-------------------------------------------------------------------------%
function [BestGen] = Harmony_Search(traindat,limits,attr)
   
    % Harmony search on illustrative dataset 
    % traindat----> matrix containing dataset without class labels (nxp)
    % UL ----> matrix with upper and lower limits for "p" attributes (px2)
    % HMCR---> Harmony Memory Consideration Rate [HMCR_max HMCR_min]
    % HMS----> Harmony Memory Size or population size
    % NVAR---> Number of attributes (p)
    % PAR---> Pitch Adjustment Rate [PAR_max PAR_min]
    
   global   ULB  HMS HMCR PAR_max PAR_min BW_min BW_max tem ;
   global   BestFit WorstFit BestGen BW HM fitness;
   global  counter BestIndex WorstIndex attributes;
   ULB = limits;                      
   BestGen =[];                             % Store the best fitness
   HMS = 5;                                 % Size of population
   NATTR = attr;                            % No. of attributes in datase              
   MaxIter = 5000;                          % Maximum no. of iterations
   
   HMCR_max = 0.95;      HMCR_min = 0.06;                  
   PAR_max = 0.95;       PAR_min = 0.35;                                                 
   BW_max = 0.01;        BW_min=0.1;                   
  
   % Initialize Matrices
   BW = zeros(1,NATTR);                     % Bandwidth values [1xp]            
   HM = zeros(HMS,NATTR) ;                  % Size of Harmony Memory (HMSxp)
   fitness = zeros(1,HMS);                  % Fitness of each population
   attributes = zeros(1,NATTR) ;     
   tem = zeros(2,NATTR) ;     
   
   HarmonySearch;
  
   % plot the final optimal cluster center
   plot(BestGen(1),BestGen(2),'k*', 'LineWidth',3) 
   legend('Agents','Agents Movement','Location','NorthEastOutside')
   xlabel('X'); ylabel('Y'); xlim([0 30]); ylim([0 35]);
   text(35,30,'* Cluster Centers', 'FontName','Times','Fontsize',12)
%    plot(BestGen(1),BestGen(2),'g*') % plot the final optimal cluster center
 % /***********************************************************************/
function objective = f(x)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % OBJECTIVE FUNCTION - Evaluation of fitness value of each solution vector
    % x receives the matrix containing the pseudo optimal cluster centres to be evaluated
    % Optimisation of Clustering Criteria by Reformulation
    % Refer Equation (4) in paper: "Optimisation of Clustering Criteria by Reformulation
    % Note: your own objective function can be written here
    % when using your own function, remember to change limits/bounds
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    objective = sum(pdist2(x,traindat)); 
    
end
% /***********************************************************************/
% Initialisation of Harmony Memory with random solutions between limits
        function initialize
              for i = 1:HMS
                for j = 1:NATTR
                    % Initialization of HM between upper and lower limit
                    HM(i,j) = ((ULB(j,2)-ULB(j,1))*rand(1))+ULB(j,1);   
                end
                fitness(i)= f(HM(i,:));    % evaluate fitness of each HM
              end
            % plot initial solutions for visualization 
            plot(HM(:,1),HM(:,2),'gs', 'LineWidth',1.5);
            hold on;
        end
% /***********************************************************************/
% Improvisation of Harmony Memory
        function HarmonySearch
           initialize;         % First intialize HM with random values
           counter = 0;        % Loop counter
           while counter < MaxIter
               % dynamically vary the Harmony Search parameters - PAR, HMCR, bw
               PAR=(PAR_max-PAR_min)/(MaxIter*counter)+PAR_min;
               coef=log(BW_min/BW_max)/MaxIter;
               for pp =1:NATTR
                  BW(pp)=BW_max*exp(coef*counter);
               end
               for hh = 1:NATTR
                  HMCR(hh) = HMCR_min+(((HMCR_max-HMCR_min)/MaxIter)*counter);
               end
               % improvise the solutions using parameters
               for i = 1:NATTR
                   hmcr_rnd = rand(1);
                   if hmcr_rnd <= HMCR(i)
                      index = randi([1,HMS],1);    % randomly pick one population from HM 
                      attributes(i) = HM(index,i);
                      par_rnd = rand(1); 
                      % if random number is less than PAR, tune solution
                      if par_rnd <= PAR
                         walk_rnd = rand(1);
                         temp = attributes(i);      
                         if walk_rnd<0.5 % tune i.e. add/subtract small value with 50% probabilty
                             temp = temp+rand(1)*BW(i);  
                             % check if improvised solution is within upper limit
                             if temp < ULB(i,1) 
                                attributes(i) = temp;
                             end         
                         else          
                             temp = temp-rand(1)*BW(i);
                             % check if improvised solution is within lower limit
                             if temp>ULB(i,2)
                                attributes(i) = temp;
                             end                            
                         end                     
                      end          % End of if for r2
                   else
                      % Randomisation with probability of 1-HMCR
                      attributes(i) = ((ULB(i,1)-ULB(i,2))*rand(1))+ULB(i,2);    
                   end  
               end
               value = f(attributes);         % calculate fitness of  improvised solution
               UpdateHM(value);        % Update the fitness value of the HM
               counter = counter+1;    % Increment loop counter
           end
        end
% /***********************************************************************/
% Update the Harmony Memory
        function UpdateHM( NewFit )
            % For the first iteration, find best & worst fitness solutions with index
            if(counter==0)
                [BestFit, BestIndex] = min(fitness);
                [WorstFit, WorstIndex] = max(fitness);
            end
            % for second iteration iteration onwards
               tem(1,:) = HM(WorstIndex,:);
               % replace if less than worst fitness
               if (NewFit < WorstFit)
                   % if lower than previous best fitness, update new fitness
                   % as the best fitness while discarding the worst fitness
                  if( NewFit < BestFit )
                     HM(WorstIndex,:)=attributes;
                     BestGen=attributes;
                     tem(2,:) = attributes;
                     fitness(WorstIndex)=NewFit;
                     BestIndex=WorstIndex;
                   % if only lower than worst fitness but not best fitness, then just
                   % replace worst fitness
                  else
                    HM(WorstIndex,:)=attributes;
                    tem(2,:) = attributes;
                    fitness(WorstIndex)=NewFit;
                  end
                  
                   % plot the movement of the solutions
                   pause(0.0005)  % Pause interval for tracing movements can be changed here     
                   hold on;
                   plot(tem(:,1),tem(:,2),'k:'); 
                   
                 % find the new worst solution and its index
                 [WorstFit, WorstIndex] = max(fitness);
              end % NewFit if
        end % function update
toc;
end      % end of function