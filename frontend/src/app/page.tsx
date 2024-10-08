import Link from "next/link";

import { getServerAuthSession } from "~/server/auth";
import { api } from "~/trpc/server";
import styles from "./index.module.css";
import { SelectMovie } from "./_components/select-movie";
import { Movie } from "~/server/db/schema";
import { CreatePost } from "./_components/create-post";
import Landing from "./_components/landing/landing";

export default async function Home() {
  const hello = await api.post.hello({ text: "from tRPC" });
  const session = await getServerAuthSession();

  return (
    <Landing/>
  );
}