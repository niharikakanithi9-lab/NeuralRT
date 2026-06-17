#include "neuralrt/model/sequential.h"
#include <gtest/gtest.h>
#include "neuralrt/ops/linear.h"

using neuralrt::Shape;
using neuralrt::Tensor;
using neuralrt::model::LinearLayer;
using neuralrt::model::ReLULayer;
using neuralrt::model::Sequential;

TEST(SequentialTest, EmptyModelReturnsInputUnchanged) {
    Sequential model;
    Tensor input(Shape({1, 3}), {1.0f, 2.0f, 3.0f});
    Tensor output = model.forward(input);
    EXPECT_FLOAT_EQ(output.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(output.at({0, 2}), 3.0f);
}

TEST(SequentialTest, NumLayersTracksAdditions) {
    Sequential model;
    model.add(std::make_unique<LinearLayer>(4, 8));
    model.add(std::make_unique<ReLULayer>());
    model.add(std::make_unique<LinearLayer>(8, 2));
    EXPECT_EQ(model.num_layers(), 3u);
}

TEST(SequentialTest, ForwardProducesCorrectOutputShape) {
    Sequential model;
    model.add(std::make_unique<LinearLayer>(4, 8));
    model.add(std::make_unique<ReLULayer>());
    model.add(std::make_unique<LinearLayer>(8, 2));
    Tensor input = Tensor::random(Shape({5, 4}));
    Tensor output = model.forward(input);
    EXPECT_EQ(output.shape().dim(0), 5u);
    EXPECT_EQ(output.shape().dim(1), 2u);
}

TEST(SequentialTest, ReLUClipsNegativesBetweenLayers) {
    Sequential model;
    Tensor weight(Shape({1, 1}), {-1.0f});
    Tensor bias(Shape({1}), {0.0f});

    class FixedLinear : public neuralrt::model::Layer {
    public:
        FixedLinear(Tensor w, Tensor b) : linear_(std::move(w), std::move(b)) {}
        Tensor forward(const Tensor& input) const override { return linear_.forward(input); }
    private:
        neuralrt::ops::Linear linear_;
    };

    model.add(std::make_unique<FixedLinear>(weight, bias));
    model.add(std::make_unique<ReLULayer>());
    Tensor input(Shape({1, 1}), {5.0f});
    Tensor output = model.forward(input);
    EXPECT_FLOAT_EQ(output.at({0, 0}), 0.0f);
}