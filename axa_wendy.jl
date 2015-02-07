using DataFrames

using StatsBase
using Gadfly
using GLM


function get_features(basepath, d, t)

    train_d1_r1 = readtable(basepath*d*"/"*string(t)*".csv",separator=',');

    diff_loc=DataArray(train_d1_r1[2:end,:])-DataArray(train_d1_r1[1:end-1,:]);
    speed=sqrt(diff_loc[:,1].^2+diff_loc[:,2].^2);

    #take moving avg on speed
    mavg_n = 5
    mavg_h = ones(Float64, mavg_n) / mavg_n
    # conv is the convolution function
    speed=conv(array(speed), mavg_h)

    acc = speed[2:end,:]-speed[1:end-1,:];
    jerk = acc[2:end,:]-acc[1:end-1,:];
    # when speed is zero


    return DataFrame(max_spd=maximum(speed), med_spd=median(speed), min_spd=minimum(speed), mean_spd=mean(speed),
        max_acc=maximum(acc), med_acc=median(acc), min_acc=minimum(acc), mean_acc=mean(acc),
        max_jrk=maximum(jerk), med_jrk=median(jerk), min_jrk=minimum(jerk), mean_jrk=mean(jerk));
end



#plot(x=collect(1:size(speed,1)), y=speed,Scale.x_continuous(format=:plain), Scale.y_continuous(format=:plain),Guide.xlabel("time"),Guide.ylabel("speed"))

# ============================================================
# MAIN
basepath = "/Users/wendy/Documents/comps/AXA/data/drivers/";
drivers = readdir(basepath);
f = open("sub1.csv","w");
write(f,"driver_trip,prob\n");
numDriverToCompare = 5;

for d in drivers
    trips=[1:200];

    # define a dataframe of features extracted from all trips of this driver
    tripsDF = DataFrame(max_spd= Float64[], med_spd= Float64[], min_spd= Float64[], mean_spd= Float64[],
        max_acc= Float64[], med_acc= Float64[], min_acc= Float64[], mean_acc= Float64[],
        max_jrk= Float64[], med_jrk= Float64[], min_jrk= Float64[], mean_jrk= Float64[]);

    for t in trips
        feat = get_features(basepath, d, t);
        tripsDF = append!(feat,tripsDF);
    end

    #randomly pick some other drivers
    for otherDriver in [1:numDriverToCompare]
        feat = get_features(basepath, drivers[rand(1:length(drivers))], rand(1:200));
        tripsDF = append!(feat,tripsDF);
    end

    # the truth is 200 ones and 5 zeros
    tripsDF[:truth]= vcat(ones(200), -ones(numDriverToCompare));
    model = fit(LinearModel, truth ~ max_spd +med_spd +min_spd  +mean_spd +max_acc  +med_acc  +min_acc +mean_acc +max_jrk  +med_jrk  +min_jrk  +mean_jrk,tripsDF);
    pred = predict(model);


    for t in trips
        write(f,@sprintf("%s_%s,%f\n",d, t, pred[t]));
    end

end
close(f)