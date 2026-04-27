import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";

export default defineConfig({
  site: "https://sol-momma.github.io",
  base: "/SpecialProject",
  integrations: [mdx()],
  output: "static",
});
