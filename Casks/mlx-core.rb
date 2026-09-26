cask "mlx-core" do
  version "26.9.6"
  sha256 "e149f4d0acdfa1312fb2ca084141e4fb878c046bb8651bafbb32736559f71269"

  url "https://github.com/ddalcu/mlx-serve/releases/download/v#{version}/MLX-Serve.dmg"
  name "MLX-Serve"
  desc "Native LLM server for Apple Silicon with OpenAI & Anthropic compatible APIs"
  homepage "https://github.com/ddalcu/mlx-serve"

  depends_on macos: :tahoe
  depends_on arch: :arm64

  app "MLX-Serve.app"

  zap trash: [
    "~/.mlx-serve",
  ]
end
