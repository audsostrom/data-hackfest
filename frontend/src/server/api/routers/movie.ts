import { z } from 'zod';
import { eq } from 'drizzle-orm';

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from '~/server/api/trpc';
import { movieProductions, movies } from '~/server/db/schema';
import { datetime } from 'drizzle-orm/mysql-core';

//** movie route */
export const movieRouter = createTRPCRouter({
  // defines purpose, protected = requires login
  byId: protectedProcedure
    // validates input
    .input(z.number())
    // sets up query with context and input
    .query(({ ctx, input }) => {
      // drizzle orm select
      // https://orm.drizzle.team/docs/rqb
      return ctx.db.query.movies.findFirst({
        where: eq(movies.id, input),
        with:{
          movieProductions: {
            with:{
              production: true,
            },
          },
          reviews: true,
        }
      })
    }),

  add: protectedProcedure
  .input(z.object({id: z.number(), title: z.string(), genres: z.string()}))
  .mutation(async ({ ctx, input }) => {
    
    await ctx.db.insert(movies).values({
      id: input.id,
      title: input.title,
      genres: input.genres,
      length: 0,
      release_dt: new Date(),
      synopsis: "",
      vote_avg: 0,
      vote_count: 0,
      lang: "en",
      mpaa_rating: "PG",
    });
  }),
  getAll: protectedProcedure.query(({ ctx }) => {
    return ctx.db.query.movies.findMany();
  }),
});
